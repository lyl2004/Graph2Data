"""Line mask skeletonization and ordered path tracing."""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .image_io import load_bgr, write_json
from .models import CurvePath, Point

Pixel = Tuple[int, int]  # (x, y)


@dataclass
class PathTracingConfig:
    threshold: int = 127
    min_component_pixels: int = 1
    prune_spurs: bool = True
    max_spur_length_px: int = 12
    prefer_longest_component: bool = False
    enable_gap_linking: bool = True
    max_gap_px: float = 45.0
    max_gap_angle_deg: float = 60.0
    max_gap_tangent_angle_deg: float = 55.0
    use_tangent_gap_interpolation: bool = False
    backward_x_tolerance_px: float = 2.0
    segment_tangent_weight: float = 0.18
    segment_next_tangent_weight: float = 0.12
    segment_curvature_weight: float = 0.03
    segment_overlap_penalty: float = 20.0
    segment_overlap_weight: float = 0.0
    segment_long_gap_weight: float = 0.0
    segment_short_length_threshold: int = 4
    segment_short_length_penalty: float = 0.0
    completed_point_confidence: float = 0.35


class LinePathExtractor:
    """Extract ordered centerline paths from binary or color masks."""

    def __init__(self, config: Optional[PathTracingConfig] = None):
        self.config = config or PathTracingConfig()

    def extract_from_mask_file(self, mask_path: str, curve_id: str = "curve") -> CurvePath:
        image = load_bgr(mask_path)
        return self.extract_from_mask_image(image, curve_id=curve_id)

    def extract_from_mask_image(self, image_bgr: np.ndarray, curve_id: str = "curve") -> CurvePath:
        binary = self._to_binary(image_bgr)
        skeleton = self._skeletonize(binary)
        pixels = _skeleton_pixels(skeleton)
        if not pixels:
            return CurvePath(curve_id=curve_id, pixel_points_ordered=[], warnings=["empty skeleton"])

        components = _connected_components(pixels)
        components = [c for c in components if len(c) >= self.config.min_component_pixels]
        if self.config.prune_spurs:
            components = [
                _prune_short_spurs(c, self.config.max_spur_length_px)
                for c in components
            ]
            components = [c for c in components if len(c) >= self.config.min_component_pixels]
        if not components:
            return CurvePath(curve_id=curve_id, pixel_points_ordered=[], warnings=["no component above threshold"])

        components.sort(key=len, reverse=True)
        selected_components = [components[0]] if self.config.prefer_longest_component else components
        selected = set().union(*selected_components)
        graph = _build_graph(selected)
        degrees = {node: len(neighbors) for node, neighbors in graph.items()}
        endpoints = [node for node, degree in degrees.items() if degree == 1]
        junctions = [node for node, degree in degrees.items() if degree >= 3]

        completed_ranges: List[Tuple[int, int]] = []
        if len(selected_components) > 1 and not self.config.prefer_longest_component:
            ordered, completed_ranges = _trace_multi_component_path(selected_components, self.config)
        else:
            ordered = _trace_main_path(graph, endpoints)
        path_points = [Point(float(x), float(y)) for x, y in ordered]
        point_confidences = _point_confidences(
            len(path_points), completed_ranges, self.config.completed_point_confidence
        )
        endpoint_points = [Point(float(x), float(y)) for x, y in endpoints]
        junction_points = [Point(float(x), float(y)) for x, y in junctions]
        length = _path_length(ordered)
        completed_count = sum(max(0, end - start + 1) for start, end in completed_ranges)
        observed_in_path = max(0, len(ordered) - completed_count)
        coverage = observed_in_path / max(len(selected), 1)
        confidence = float(max(0.0, min(1.0, coverage)))
        warnings = []
        if len(components) > 1:
            warnings.append(f"multiple_components={len(components)}")
            if not self.config.prefer_longest_component:
                warnings.append("components_ordered_by_x")
        if junctions:
            warnings.append(f"junctions={len(junctions)}")
        if len(ordered) < len(selected):
            warnings.append("ordered path does not cover all skeleton pixels")
        if completed_ranges:
            warnings.append(f"gap_linked_points={completed_count}")

        return CurvePath(
            curve_id=curve_id,
            pixel_points_ordered=path_points,
            completed_ranges=[(int(start), int(end)) for start, end in completed_ranges],
            confidence_per_point=point_confidences,
            endpoints=endpoint_points,
            junctions=junction_points,
            component_count=len(components),
            observed_pixel_count=len(selected),
            completed_pixel_count=int(completed_count),
            path_length_px=float(length),
            confidence=confidence,
            warnings=warnings,
        )

    def _to_binary(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.config.threshold, 255, cv2.THRESH_BINARY)
        return binary

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        mask = (binary > 0).astype(np.uint8)
        try:
            from skimage.morphology import skeletonize

            return skeletonize(mask).astype(np.uint8) * 255
        except Exception:
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                return cv2.ximgproc.thinning(mask * 255)
            return _morphological_skeleton(mask)


def _skeleton_pixels(skeleton: np.ndarray) -> Set[Pixel]:
    ys, xs = np.where(skeleton > 0)
    return {(int(x), int(y)) for x, y in zip(xs, ys)}


def _connected_components(pixels: Set[Pixel]) -> List[Set[Pixel]]:
    remaining = set(pixels)
    components: List[Set[Pixel]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in _neighbors(node):
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _build_graph(pixels: Set[Pixel]) -> Dict[Pixel, List[Pixel]]:
    graph: Dict[Pixel, List[Pixel]] = {}
    for pixel in pixels:
        graph[pixel] = [nb for nb in _neighbors(pixel) if nb in pixels]
    return graph


def _prune_short_spurs(pixels: Set[Pixel], max_spur_length: int) -> Set[Pixel]:
    """Remove short endpoint branches that terminate at a junction."""
    if max_spur_length <= 0 or len(pixels) < 4:
        return set(pixels)

    pruned = set(pixels)
    changed = True
    while changed:
        changed = False
        graph = _build_orthogonal_graph(pruned)
        endpoints = [node for node, neighbors in graph.items() if len(neighbors) == 1]
        junctions = {node for node, neighbors in graph.items() if len(neighbors) >= 3}
        if not endpoints or not junctions:
            break

        to_remove: Set[Pixel] = set()
        for endpoint in endpoints:
            branch = _trace_endpoint_branch(graph, endpoint, junctions, max_spur_length)
            if branch:
                to_remove.update(branch)
        if to_remove and len(pruned) - len(to_remove) >= 2:
            pruned.difference_update(to_remove)
            changed = True
    return pruned


def _trace_endpoint_branch(
    graph: Dict[Pixel, List[Pixel]],
    endpoint: Pixel,
    junctions: Set[Pixel],
    max_spur_length: int,
) -> List[Pixel]:
    branch = [endpoint]
    previous: Optional[Pixel] = None
    current = endpoint
    while len(branch) <= max_spur_length:
        neighbors = [node for node in graph[current] if node != previous]
        if len(neighbors) != 1:
            return []
        next_node = neighbors[0]
        if next_node in junctions:
            direction = (float(next_node[0] - current[0]), float(next_node[1] - current[1]))
            continuations = [node for node in graph[next_node] if node != current]
            has_straight_continuation = any(
                _angle_between(direction, (float(node[0] - next_node[0]), float(node[1] - next_node[1]))) <= 45.0
                for node in continuations
            )
            return [] if has_straight_continuation else branch
        previous = current
        current = next_node
        branch.append(current)
    return []


def _build_orthogonal_graph(pixels: Set[Pixel]) -> Dict[Pixel, List[Pixel]]:
    graph: Dict[Pixel, List[Pixel]] = {}
    for pixel in pixels:
        graph[pixel] = [nb for nb in _orthogonal_neighbors(pixel) if nb in pixels]
    return graph


def _orthogonal_neighbors(pixel: Pixel) -> Iterable[Pixel]:
    x, y = pixel
    yield (x - 1, y)
    yield (x + 1, y)
    yield (x, y - 1)
    yield (x, y + 1)


def _neighbors(pixel: Pixel) -> Iterable[Pixel]:
    x, y = pixel
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield (x + dx, y + dy)


def _trace_main_path(graph: Dict[Pixel, List[Pixel]], endpoints: Sequence[Pixel]) -> List[Pixel]:
    if not graph:
        return []
    if len(graph) == 1:
        return [next(iter(graph))]

    if len(endpoints) >= 2:
        best_path: List[Pixel] = []
        endpoint_list = list(endpoints)
        for i, start in enumerate(endpoint_list):
            distances, parents = _bfs_tree(graph, start)
            for end in endpoint_list[i + 1 :]:
                if end in distances and distances[end] > len(best_path):
                    best_path = _reconstruct_path(parents, start, end)
        if best_path:
            return best_path

    # Closed loop or no reliable endpoints: trace from an arbitrary node to farthest, then farthest again.
    start = next(iter(graph))
    distances, _ = _bfs_tree(graph, start)
    farthest = max(distances, key=distances.get)
    distances2, parents2 = _bfs_tree(graph, farthest)
    farthest2 = max(distances2, key=distances2.get)
    return _reconstruct_path(parents2, farthest, farthest2)


def _trace_multi_component_path(components: Sequence[Set[Pixel]], config: PathTracingConfig) -> Tuple[List[Pixel], List[Tuple[int, int]]]:
    """Trace each component and concatenate component paths from left to right.

    This is a baseline for common scientific curves that can be treated as y=f(x).
    It intentionally keeps gap jumps visible in the returned polyline; later stages
    can use those gaps for dashed-line reconstruction and confidence scoring.
    """
    traced = []
    for component in components:
        graph = _build_graph(component)
        endpoints = [node for node, neighbors in graph.items() if len(neighbors) == 1]
        path = _trace_main_path(graph, endpoints)
        if not path:
            continue
        if path[0][0] > path[-1][0]:
            path = list(reversed(path))
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        traced.append(
            {
                "path": path,
                "length": len(path),
                "min_x": min(xs),
                "max_x": max(xs),
                "center_x": sum(xs) / len(xs),
                "center_y": sum(ys) / len(ys),
                "start": path[0],
                "end": path[-1],
                "start_tangent": _path_tangent(path, at_start=True),
                "end_tangent": _path_tangent(path, at_start=False),
            }
        )

    traced = _order_segments(traced, config)
    ordered: List[Pixel] = []
    completed_ranges: List[Tuple[int, int]] = []
    previous_item = None
    for item in traced:
        path = item["path"]
        if ordered:
            prev_tangent = previous_item["end_tangent"] if previous_item else None
            gap = _interpolate_gap(
                ordered[-1],
                path[0],
                config,
                prev_tangent=prev_tangent,
                next_tangent=item["start_tangent"],
            )
            if gap:
                start = len(ordered)
                ordered.extend(gap)
                completed_ranges.append((start, len(ordered) - 1))
            if ordered[-1] == path[0]:
                ordered.extend(path[1:])
            else:
                ordered.extend(path)
        else:
            ordered.extend(path)
        previous_item = item
    return ordered, completed_ranges


def _order_segments(segments: List[Dict], config: PathTracingConfig) -> List[Dict]:
    if not segments:
        return []
    remaining = sorted(segments, key=lambda item: (item["min_x"], item["center_y"], item["max_x"]))
    ordered = [remaining.pop(0)]

    while remaining:
        current = ordered[-1]
        best_idx = None
        best_cost = float("inf")
        for idx, candidate in enumerate(remaining):
            cost = _segment_connection_cost(current, candidate, config)
            if cost is None:
                continue
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx is None:
            best_idx = min(range(len(remaining)), key=lambda i: (remaining[i]["min_x"], remaining[i]["center_y"]))
        ordered.append(remaining.pop(best_idx))
    return ordered


def _segment_connection_cost(current: Dict, candidate: Dict, config: PathTracingConfig) -> Optional[float]:
    """Score how plausible it is to connect current -> candidate.

    Lower is better. Returning None means the candidate moves too far backward
    in x for the current y=f(x) baseline.
    """
    dx = float(candidate["start"][0] - current["end"][0])
    if dx < -config.backward_x_tolerance_px:
        return None
    dy = float(candidate["start"][1] - current["end"][1])
    dist = math.hypot(dx, dy)
    gap_vector = (dx, dy)
    tangent_angle = _angle_between(current["end_tangent"], gap_vector)
    next_angle = _angle_between(gap_vector, candidate["start_tangent"])
    curvature_angle = _angle_between(current["end_tangent"], candidate["start_tangent"])
    overlap = max(0.0, float(current["max_x"] - candidate["min_x"]))
    long_gap = max(0.0, dist - config.max_gap_px)
    short_len = max(0, int(config.segment_short_length_threshold) - int(candidate.get("length", len(candidate.get("path", [])))))

    cost = (
        dist
        + config.segment_tangent_weight * tangent_angle
        + config.segment_next_tangent_weight * next_angle
        + config.segment_curvature_weight * curvature_angle
        + config.segment_overlap_weight * overlap
        + config.segment_long_gap_weight * long_gap
        + config.segment_short_length_penalty * short_len
    )
    if overlap > 0:
        cost += config.segment_overlap_penalty
    return float(cost)


def _interpolate_gap(
    a: Pixel,
    b: Pixel,
    config: PathTracingConfig,
    prev_tangent: Optional[Tuple[float, float]] = None,
    next_tangent: Optional[Tuple[float, float]] = None,
) -> List[Pixel]:
    if not config.enable_gap_linking:
        return []
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist <= 1.5 or dist > config.max_gap_px:
        return []
    if dx < -1:
        return []
    angle = abs(math.degrees(math.atan2(dy, dx if dx != 0 else 1e-9)))
    if angle > config.max_gap_angle_deg:
        return []
    if prev_tangent is not None and _angle_between(prev_tangent, (dx, dy)) > config.max_gap_tangent_angle_deg:
        return []
    if next_tangent is not None and _angle_between((dx, dy), next_tangent) > config.max_gap_tangent_angle_deg:
        return []

    steps = int(round(dist))
    if steps <= 1:
        return []
    points = _interpolate_gap_points(
        a,
        b,
        steps,
        prev_tangent=prev_tangent,
        next_tangent=next_tangent,
        use_tangent=config.use_tangent_gap_interpolation,
    )
    return points


def _interpolate_gap_points(
    a: Pixel,
    b: Pixel,
    steps: int,
    prev_tangent: Optional[Tuple[float, float]] = None,
    next_tangent: Optional[Tuple[float, float]] = None,
    use_tangent: bool = True,
) -> List[Pixel]:
    if steps <= 1:
        return []
    use_hermite = use_tangent and prev_tangent is not None and next_tangent is not None
    if use_hermite:
        return _interpolate_gap_hermite(a, b, steps, prev_tangent, next_tangent)
    return _interpolate_gap_linear(a, b, steps)


def _interpolate_gap_linear(a: Pixel, b: Pixel, steps: int) -> List[Pixel]:
    points: List[Pixel] = []
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    for i in range(1, steps):
        t = i / steps
        x = int(round(a[0] + dx * t))
        y = int(round(a[1] + dy * t))
        p = (x, y)
        if p != a and p != b and (not points or points[-1] != p):
            points.append(p)
    return points


def _interpolate_gap_hermite(
    a: Pixel,
    b: Pixel,
    steps: int,
    prev_tangent: Tuple[float, float],
    next_tangent: Tuple[float, float],
) -> List[Pixel]:
    dist = math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))
    m0 = _scaled_tangent(prev_tangent, dist)
    m1 = _scaled_tangent(next_tangent, dist)
    points: List[Pixel] = []
    for i in range(1, steps):
        t = i / steps
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        x = int(round(h00 * a[0] + h10 * m0[0] + h01 * b[0] + h11 * m1[0]))
        y = int(round(h00 * a[1] + h10 * m0[1] + h01 * b[1] + h11 * m1[1]))
        p = (x, y)
        if p != a and p != b and (not points or points[-1] != p):
            points.append(p)
    return points


def _scaled_tangent(tangent: Tuple[float, float], gap_distance: float) -> Tuple[float, float]:
    tx, ty = tangent
    norm = math.hypot(tx, ty)
    if norm == 0:
        return (0.0, 0.0)
    scale = gap_distance / norm
    return (tx * scale, ty * scale)


def _path_tangent(path: Sequence[Pixel], at_start: bool, window: int = 5) -> Tuple[float, float]:
    if len(path) < 2:
        return (1.0, 0.0)
    if at_start:
        a = path[0]
        b = path[min(window, len(path) - 1)]
    else:
        a = path[max(0, len(path) - 1 - window)]
        b = path[-1]
    return (float(b[0] - a[0]), float(b[1] - a[1]))


def _angle_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_v = max(-1.0, min(1.0, (ax * bx + ay * by) / (norm_a * norm_b)))
    return abs(math.degrees(math.acos(cos_v)))


def _point_confidences(length: int, completed_ranges: Sequence[Tuple[int, int]], completed_confidence: float) -> List[float]:
    confidences = [1.0] * length
    for start, end in completed_ranges:
        for idx in range(max(0, start), min(length, end + 1)):
            confidences[idx] = float(completed_confidence)
    return confidences


def _bfs_tree(graph: Dict[Pixel, List[Pixel]], start: Pixel):
    distances = {start: 0}
    parents: Dict[Pixel, Optional[Pixel]] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb in graph[node]:
            if nb not in distances:
                distances[nb] = distances[node] + 1
                parents[nb] = node
                queue.append(nb)
    return distances, parents


def _reconstruct_path(parents: Dict[Pixel, Optional[Pixel]], start: Pixel, end: Pixel) -> List[Pixel]:
    path = [end]
    node = end
    while node != start:
        parent = parents.get(node)
        if parent is None:
            return []
        node = parent
        path.append(node)
    path.reverse()
    return path


def _path_length(points: Sequence[Pixel]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
    return total


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = mask.copy() * 255
    while True:
        eroded = cv2.erode(work, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, temp)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded.copy()
        if cv2.countNonZero(work) == 0:
            break
    return skel


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an ordered path from a curve mask.")
    parser.add_argument("--mask", required=True, help="Input binary/color mask path")
    parser.add_argument("--curve_id", default="curve", help="Curve id for the output")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument("--max_gap", type=float, default=45.0, help="Maximum gap length for linking")
    parser.add_argument("--max_gap_angle", type=float, default=60.0, help="Maximum angle mismatch for gap linking")
    parser.add_argument("--max_gap_tangent_angle", type=float, default=55.0, help="Maximum tangent mismatch for gap linking")
    parser.add_argument("--tangent_gap_interpolation", action="store_true", help="Use tangent-guided Hermite interpolation for linked gaps")
    parser.add_argument("--min_component_pixels", type=int, default=1, help="Minimum skeleton component size")
    parser.add_argument("--max_spur_length", type=int, default=12, help="Maximum endpoint spur length to prune before path tracing")
    parser.add_argument("--no_spur_pruning", action="store_true", help="Disable short spur pruning")
    parser.add_argument("--no_gap_linking", action="store_true", help="Disable gap interpolation")
    args = parser.parse_args()

    config = PathTracingConfig(
        enable_gap_linking=not args.no_gap_linking,
        min_component_pixels=args.min_component_pixels,
        prune_spurs=not args.no_spur_pruning,
        max_spur_length_px=args.max_spur_length,
        max_gap_px=args.max_gap,
        max_gap_angle_deg=args.max_gap_angle,
        max_gap_tangent_angle_deg=args.max_gap_tangent_angle,
        use_tangent_gap_interpolation=args.tangent_gap_interpolation,
    )
    result = LinePathExtractor(config).extract_from_mask_file(args.mask, curve_id=args.curve_id)
    if args.out:
        write_json(args.out, result)
    else:
        import json

        from .models import to_serializable

        print(json.dumps(to_serializable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

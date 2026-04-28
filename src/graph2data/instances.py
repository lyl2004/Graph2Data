"""Curve instance diagnostics built from marker candidates."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .models import LineComponentClassification, LineStyleCurveInstance, MarkerCandidate, MarkerCurveInstance, Point


def should_group_marker_curve_instances(
    marker_candidates: Sequence[MarkerCandidate],
    line_components: Sequence[object] | None = None,
) -> bool:
    if not marker_candidates:
        return False
    if line_components:
        # Dashed/dotted line fragments often look compact enough to be marker
        # candidates. If skeleton component fragmentation is higher than the
        # marker candidate count, treat this as a line-style scene for now.
        if len(line_components) > len(marker_candidates):
            return False
    return True


def group_marker_curve_instances(
    marker_candidates: Sequence[MarkerCandidate],
    group_count: int | None = None,
    x_tolerance_px: float = 14.0,
    prefer_trajectory: bool = False,
) -> List[MarkerCurveInstance]:
    if not marker_candidates:
        return []
    candidates = list(marker_candidates)
    count = group_count or _infer_group_count(candidates)
    groups = (
        cluster_markers_by_trajectory(candidates, count)
        if prefer_trajectory
        else cluster_markers_by_x_rank(candidates, count, x_tolerance_px=x_tolerance_px)
    )
    if not groups["centers"]:
        groups = cluster_markers_by_y(candidates, count)
    instances: List[MarkerCurveInstance] = []
    for group_idx, marker_indices in sorted(groups["members"].items()):
        markers = [candidates[idx] for idx in marker_indices]
        if not markers:
            continue
        markers.sort(key=lambda marker: (marker.center.x, marker.center.y))
        xs = [marker.center.x for marker in markers]
        ys = [marker.center.y for marker in markers]
        source_curve_ids = sorted({marker.curve_id for marker in markers})
        source_curve_id = source_curve_ids[0] if len(source_curve_ids) == 1 else "mixed"
        instances.append(
            MarkerCurveInstance(
                instance_id=f"marker_instance_{len(instances):02d}",
                source_curve_id=source_curve_id,
                marker_ids=[marker.marker_id for marker in markers],
                points=[marker.center for marker in markers],
                marker_count=len(markers),
                center_y=sum(ys) / len(ys),
                x_min=min(xs),
                x_max=max(xs),
                y_min=min(ys),
                y_max=max(ys),
                grouping_method=groups.get("method", "unknown"),
                confidence=min(1.0, len(markers) / 8.0),
                warnings=[] if source_curve_id != "mixed" else ["mixed_source_curve_ids"],
            )
        )
    return instances


def should_group_line_style_curve_instances(
    line_components: Sequence[LineComponentClassification],
    group_count: int | None = None,
) -> bool:
    line_like = [component for component in line_components if component.class_label == "line_like"]
    if group_count is None or group_count <= 1:
        return False
    return len(line_like) >= group_count


def filter_border_line_components(
    line_components: Sequence[LineComponentClassification],
    plot_bbox,
    edge_tolerance_px: float = 10.0,
) -> List[LineComponentClassification]:
    filtered = []
    plot_width = max(1.0, float(plot_bbox.width))
    for component in line_components:
        if component.class_label == "line_like":
            near_top = abs(float(component.bbox.y_min) - float(plot_bbox.y_min)) <= edge_tolerance_px
            near_bottom = abs(float(component.bbox.y_max) - float(plot_bbox.y_max)) <= edge_tolerance_px
            long_horizontal = component.width >= plot_width * 0.20 and component.height <= 8.0
            if long_horizontal and (near_top or near_bottom):
                continue
        filtered.append(component)
    return filtered


def group_line_style_curve_instances(
    line_components: Sequence[LineComponentClassification],
    group_count: int,
) -> List[LineStyleCurveInstance]:
    line_like = [component for component in line_components if component.class_label == "line_like"]
    if not line_like or group_count <= 0:
        return []
    groups = cluster_components_by_x_rank(line_like, group_count)
    if not groups["centers"]:
        groups = cluster_components_by_y(line_like, group_count)
    instances: List[LineStyleCurveInstance] = []
    for group_idx, component_indices in sorted(groups["members"].items()):
        components = [line_like[idx] for idx in component_indices]
        if not components:
            continue
        components, outlier_count = _trim_component_y_outliers(components)
        components.sort(key=lambda component: (component.bbox.x_min, component.bbox.y_min))
        xs_min = [component.bbox.x_min for component in components]
        xs_max = [component.bbox.x_max for component in components]
        ys_min = [component.bbox.y_min for component in components]
        ys_max = [component.bbox.y_max for component in components]
        center_y = sum((component.bbox.y_min + component.bbox.y_max) / 2.0 for component in components) / len(components)
        source_curve_ids = sorted({component.curve_id for component in components})
        source_curve_id = source_curve_ids[0] if len(source_curve_ids) == 1 else "mixed"
        estimated_style = _estimate_line_style(components)
        points = []
        for component in components:
            if component.path_points:
                component_points = sorted(component.path_points, key=lambda point: (point.x, point.y))
                points.extend(component_points)
            else:
                y_center = float((component.bbox.y_min + component.bbox.y_max) / 2.0)
                x0 = float(component.bbox.x_min)
                x1 = float(component.bbox.x_max)
                if abs(x1 - x0) <= 1.0:
                    points.append(Point(x0, y_center))
                else:
                    points.append(Point(x0, y_center))
                    points.append(Point(x1, y_center))
        warnings = [] if source_curve_id != "mixed" else ["mixed_source_curve_ids"]
        if outlier_count:
            warnings.append(f"line_component_y_outliers_removed={outlier_count}")
        instances.append(
            LineStyleCurveInstance(
                instance_id=f"line_style_instance_{len(instances):02d}",
                source_curve_id=source_curve_id,
                component_ids=[component.component_id for component in components],
                points=points,
                component_count=len(components),
                center_y=float(center_y),
                x_min=min(xs_min),
                x_max=max(xs_max),
                y_min=min(ys_min),
                y_max=max(ys_max),
                estimated_line_style=estimated_style,
                grouping_method=groups.get("method", "component_y_rank"),
                confidence=min(1.0, len(components) / 6.0),
                warnings=warnings,
            )
        )
    return instances


def cluster_components_by_x_rank(
    line_components: Sequence[LineComponentClassification],
    group_count: int,
    x_tolerance_px: float = 70.0,
) -> Dict:
    k = min(group_count, len(line_components))
    if k <= 0:
        return {"centers": [], "assignments": {}, "members": {}, "method": "component_x_rank"}

    assignments: Dict[int, int] = {}
    components = list(line_components)
    centers = [
        (
            float((component.bbox.x_min + component.bbox.x_max) / 2.0),
            float((component.bbox.y_min + component.bbox.y_max) / 2.0),
        )
        for component in components
    ]
    for idx, (cx, _) in enumerate(centers):
        neighbors = [
            (other_idx, centers[other_idx])
            for other_idx in range(len(components))
            if abs(centers[other_idx][0] - cx) <= x_tolerance_px
        ]
        if len(neighbors) < 2:
            continue
        neighbors.sort(key=lambda item: item[1][1])
        rank_positions = [pos for pos, (other_idx, _) in enumerate(neighbors) if other_idx == idx]
        if not rank_positions:
            continue
        rank = rank_positions[0]
        if len(neighbors) != k:
            rank = int(round(rank * max(k - 1, 0) / max(len(neighbors) - 1, 1)))
        assignments[idx] = max(0, min(k - 1, rank))

    if len(assignments) < max(1, len(components) // 3):
        return {"centers": [], "assignments": {}, "members": {}, "method": "component_x_rank"}

    members = {idx: [] for idx in range(k)}
    for component_idx, group_idx in assignments.items():
        members[group_idx].append(component_idx)
    y_centers = []
    for group_idx in range(k):
        indices = members[group_idx]
        if indices:
            y_centers.append(sum(centers[idx][1] for idx in indices) / len(indices))
        else:
            y_centers.append(0.0)
    return {"centers": y_centers, "assignments": assignments, "members": members, "method": "component_x_rank"}


def _trim_component_y_outliers(
    components: Sequence[LineComponentClassification],
    max_distance_px: float = 130.0,
) -> tuple[List[LineComponentClassification], int]:
    if len(components) < 4:
        return list(components), 0
    centers = [
        float((component.bbox.y_min + component.bbox.y_max) / 2.0)
        for component in components
    ]
    ordered = sorted(centers)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    kept = [
        component
        for component, center_y in zip(components, centers)
        if abs(center_y - median) <= max_distance_px
    ]
    if len(kept) < max(2, len(components) // 3):
        return list(components), 0
    return kept, len(components) - len(kept)


def cluster_components_by_y(line_components: Sequence[LineComponentClassification], group_count: int, max_iterations: int = 20) -> Dict:
    k = min(group_count, len(line_components))
    if k <= 0:
        return {"centers": [], "assignments": {}, "members": {}, "method": "component_y_kmeans"}
    ys = [float((component.bbox.y_min + component.bbox.y_max) / 2.0) for component in line_components]
    weights = [max(1.0, float(component.path_length_px or component.pixel_count)) for component in line_components]
    sorted_ys = sorted(ys)
    centers = [sorted_ys[min(len(sorted_ys) - 1, int(round((i + 0.5) * len(sorted_ys) / k - 0.5)))] for i in range(k)]
    assignments: Dict[int, int] = {}
    members: Dict[int, List[int]] = {idx: [] for idx in range(k)}
    for _ in range(max_iterations):
        changed = False
        members = {idx: [] for idx in range(k)}
        for component_idx, y in enumerate(ys):
            group_idx = min(range(k), key=lambda idx: abs(y - centers[idx]))
            if assignments.get(component_idx) != group_idx:
                changed = True
            assignments[component_idx] = group_idx
            members[group_idx].append(component_idx)
        for group_idx, indices in members.items():
            if indices:
                weighted_sum = sum(ys[idx] * weights[idx] for idx in indices)
                total_weight = sum(weights[idx] for idx in indices)
                centers[group_idx] = weighted_sum / max(total_weight, 1.0)
        if not changed:
            break
    order = sorted(range(k), key=lambda idx: centers[idx])
    remap = {old: new for new, old in enumerate(order)}
    remapped_assignments = {idx: remap[group_idx] for idx, group_idx in assignments.items()}
    remapped_members = {remap[group_idx]: indices for group_idx, indices in members.items()}
    remapped_centers = [centers[old] for old in order]
    return {"centers": remapped_centers, "assignments": remapped_assignments, "members": remapped_members, "method": "component_y_kmeans"}


def cluster_markers_by_y(marker_candidates: Sequence[MarkerCandidate], group_count: int, max_iterations: int = 20) -> Dict:
    k = min(group_count, len(marker_candidates))
    ys = [float(marker.center.y) for marker in marker_candidates]
    if k <= 0:
        return {"centers": [], "assignments": {}, "members": {}, "method": "y_kmeans"}
    sorted_ys = sorted(ys)
    centers = [sorted_ys[min(len(sorted_ys) - 1, int(round((i + 0.5) * len(sorted_ys) / k - 0.5)))] for i in range(k)]
    assignments: Dict[int, int] = {}
    members: Dict[int, List[int]] = {idx: [] for idx in range(k)}
    for _ in range(max_iterations):
        changed = False
        members = {idx: [] for idx in range(k)}
        for marker_idx, y in enumerate(ys):
            group_idx = min(range(k), key=lambda idx: abs(y - centers[idx]))
            if assignments.get(marker_idx) != group_idx:
                changed = True
            assignments[marker_idx] = group_idx
            members[group_idx].append(marker_idx)
        for group_idx, indices in members.items():
            if indices:
                centers[group_idx] = sum(ys[idx] for idx in indices) / len(indices)
        if not changed:
            break
    order = sorted(range(k), key=lambda idx: centers[idx])
    remap = {old: new for new, old in enumerate(order)}
    remapped_assignments = {idx: remap[group_idx] for idx, group_idx in assignments.items()}
    remapped_members = {remap[group_idx]: indices for group_idx, indices in members.items()}
    remapped_centers = [centers[old] for old in order]
    return {"centers": remapped_centers, "assignments": remapped_assignments, "members": remapped_members, "method": "y_kmeans"}


def cluster_markers_by_x_rank(marker_candidates: Sequence[MarkerCandidate], group_count: int, x_tolerance_px: float = 14.0) -> Dict:
    k = min(group_count, len(marker_candidates))
    if k <= 0:
        return {"centers": [], "assignments": {}, "members": {}, "method": "x_rank"}

    assignments: Dict[int, int] = {}
    markers = list(marker_candidates)
    for idx, marker in enumerate(markers):
        neighbors = [
            (other_idx, other)
            for other_idx, other in enumerate(markers)
            if abs(float(other.center.x) - float(marker.center.x)) <= x_tolerance_px
        ]
        if len(neighbors) < 2:
            continue
        neighbors.sort(key=lambda item: float(item[1].center.y))
        rank_positions = [pos for pos, (other_idx, _) in enumerate(neighbors) if other_idx == idx]
        if not rank_positions:
            continue
        rank = rank_positions[0]
        if len(neighbors) != k:
            rank = int(round(rank * max(k - 1, 0) / max(len(neighbors) - 1, 1)))
        assignments[idx] = max(0, min(k - 1, rank))

    if len(assignments) < max(1, len(markers) // 3):
        return {"centers": [], "assignments": {}, "members": {}, "method": "x_rank"}

    members = {idx: [] for idx in range(k)}
    for marker_idx, group_idx in assignments.items():
        members[group_idx].append(marker_idx)
    centers = []
    for group_idx in range(k):
        indices = members[group_idx]
        if indices:
            centers.append(sum(float(markers[idx].center.y) for idx in indices) / len(indices))
        else:
            centers.append(0.0)
    return {"centers": centers, "assignments": assignments, "members": members, "method": "x_rank"}


def cluster_markers_by_trajectory(
    marker_candidates: Sequence[MarkerCandidate],
    group_count: int,
    min_inliers: int = 6,
    inlier_residual_px: float = 10.0,
    assign_residual_px: float = 22.0,
) -> Dict:
    """Cluster marker candidates by approximately linear curve trajectories.

    Vertical rank works for separated marker curves, but it swaps identities
    when curves cross. This deterministic RANSAC-like pass is reserved for
    scenes where the caller has already detected line fragmentation that
    suggests crossings.
    """
    k = min(group_count, len(marker_candidates))
    if k <= 0:
        return {"centers": [], "assignments": {}, "members": {}, "method": "trajectory_ransac"}
    markers = list(marker_candidates)
    points = [(float(marker.center.x), float(marker.center.y), idx) for idx, marker in enumerate(markers)]
    x_values = [point[0] for point in points]
    x_span = max(x_values) - min(x_values) if x_values else 0.0
    if x_span <= 1.0:
        return {"centers": [], "assignments": {}, "members": {}, "method": "trajectory_ransac"}

    min_pair_span = max(80.0, x_span * 0.30)
    candidate_lines = []
    for left_idx, left in enumerate(points):
        for right in points[left_idx + 1 :]:
            if abs(right[0] - left[0]) < min_pair_span:
                continue
            line = _line_from_points(left, right)
            if line is None:
                continue
            inliers = [
                idx
                for x_val, y_val, idx in points
                if _line_residual_px(line, x_val, y_val) <= inlier_residual_px
            ]
            if len(inliers) >= min_inliers:
                candidate_lines.append((len(inliers), line, inliers))
    candidate_lines.sort(key=lambda item: item[0], reverse=True)

    selected_lines = []
    covered = set()
    for _, line, inliers in candidate_lines:
        new_inliers = [idx for idx in inliers if idx not in covered]
        if len(new_inliers) < min_inliers:
            continue
        selected_lines.append(line)
        covered.update(new_inliers)
        if len(selected_lines) >= k:
            break
    if len(selected_lines) < k:
        return {"centers": [], "assignments": {}, "members": {}, "method": "trajectory_ransac"}

    for _ in range(2):
        provisional = {idx: [] for idx in range(k)}
        for x_val, y_val, marker_idx in points:
            best_group = min(range(k), key=lambda group_idx: _line_residual_px(selected_lines[group_idx], x_val, y_val))
            if _line_residual_px(selected_lines[best_group], x_val, y_val) <= assign_residual_px:
                provisional[best_group].append(marker_idx)
        for group_idx, marker_indices in provisional.items():
            if len(marker_indices) >= 2:
                selected_lines[group_idx] = _fit_line_to_markers([markers[idx] for idx in marker_indices])

    members = {idx: [] for idx in range(k)}
    assignments: Dict[int, int] = {}
    for x_val, y_val, marker_idx in points:
        best_group = min(range(k), key=lambda group_idx: _line_residual_px(selected_lines[group_idx], x_val, y_val))
        assignments[marker_idx] = best_group
        members[best_group].append(marker_idx)

    if any(not members[group_idx] for group_idx in range(k)):
        return {"centers": [], "assignments": {}, "members": {}, "method": "trajectory_ransac"}
    centers = [
        sum(float(markers[idx].center.y) for idx in members[group_idx]) / len(members[group_idx])
        for group_idx in range(k)
    ]
    return {"centers": centers, "assignments": assignments, "members": members, "method": "trajectory_ransac"}


def _line_from_points(left, right):
    dx = float(right[0]) - float(left[0])
    if abs(dx) <= 1e-6:
        return None
    slope = (float(right[1]) - float(left[1])) / dx
    intercept = float(left[1]) - slope * float(left[0])
    return slope, intercept


def _fit_line_to_markers(markers: Sequence[MarkerCandidate]):
    xs = [float(marker.center.x) for marker in markers]
    ys = [float(marker.center.y) for marker in markers]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x_val - x_mean) ** 2 for x_val in xs)
    if denom <= 1e-9:
        return 0.0, y_mean
    slope = sum((x_val - x_mean) * (y_val - y_mean) for x_val, y_val in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def _line_residual_px(line, x_val: float, y_val: float) -> float:
    slope, intercept = line
    return abs(slope * float(x_val) + intercept - float(y_val)) / max((slope * slope + 1.0) ** 0.5, 1e-9)


def _infer_group_count(marker_candidates: Sequence[MarkerCandidate]) -> int:
    if len(marker_candidates) >= 16:
        return min(8, max(2, round(len(marker_candidates) / 18)))
    return 1


def _estimate_line_style(components: Sequence[LineComponentClassification]) -> str:
    if not components:
        return "unknown"
    long_components = sum(1 for component in components if component.max_span >= 70 or component.path_length_px >= 80)
    short_components = sum(1 for component in components if component.max_span < 35)
    if long_components >= max(1, len(components) // 3):
        return "solid"
    if len(components) >= 8 and short_components >= len(components) * 0.6:
        return "dotted"
    if len(components) >= 3:
        return "dashed"
    return "unknown"

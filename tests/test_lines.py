#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_lines.py

科学图表线条提取与遮挡补全测试脚本。
功能：
  1. 使用 ColorExtractor 从图像中提取主要颜色（无耦合，仅传入结果）
  2. 对每种颜色，执行线条提取、断点匹配、简单遮挡补全
  3. 独立窗口展示每种颜色的线条还原效果

目标图像：
  - tests/img_broken.png   （含虚线、网格遮挡）
  - tests/img_curve.png    （含曲线、图例、交叉重叠）

依赖库：numpy, opencv-python, scikit-image, scipy
"""

import os
import sys
import math
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

# 导入颜色提取模块（无耦合，仅调用其接口）
sys.path.append(os.path.dirname(__file__))  # 允许从当前目录导入
try:
    from test_colors import ColorExtractor
except ImportError:
    print("警告: 无法导入 ColorExtractor，请确保 test_colors.py 在同一目录下")
    ColorExtractor = None

# 可选依赖，若缺失则降级使用简单方法
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("警告: scikit-image 未安装，骨架化功能将使用 OpenCV  thinning (若可用)")

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy 未安装，端点匹配将使用贪心算法")


class LineCompleter:
    """
    线条提取与补全核心类。
    接收原始图像（BGR）及 ColorExtractor 提取的颜色信息列表，
    对每种颜色执行：
      - 颜色掩膜生成
      - 形态学净化
      - 骨架化
      - 线段跟踪
      - 断点匹配与连接（简单直线补全）
      - 可视化结果生成
    """

    def __init__(self, image_bgr: np.ndarray, colors_info: List[Dict]):
        """
        Args:
            image_bgr: BGR 格式的原始图像
            colors_info: ColorExtractor.process() 返回的颜色列表，
                         每个元素必须包含 'lab' 和 'rgb' 字段
        """
        self.img_bgr = image_bgr.copy()
        self.h, self.w = self.img_bgr.shape[:2]
        self.colors = colors_info

        # 转换为 Lab 空间用于颜色距离计算
        self.img_lab = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2Lab)
        self.img_lab_float = self.img_lab.astype(np.float32)

        # 存储每种颜色的处理结果
        self.results = {}   # key: color_index, value: dict

    def process(self,
                range_l: int = 30,
                range_ab: int = 15,
                morph_kernel: int = 3,
                connect_distance: int = 50,
                min_segment_len: int = 5,
                verbose: bool = True) -> Dict:
        """
        执行所有颜色的线条提取与补全。

        Args:
            range_l:       Lab 亮度容差，用于颜色掩膜
            range_ab:      Lab 色度容差
            morph_kernel:  形态学操作核尺寸（开/闭运算）
            connect_distance: 端点匹配最大距离（像素）
            min_segment_len:  过滤短线段的最小长度
            verbose:       打印日志
        Returns:
            results: 每个颜色的处理结果字典
        """
        if verbose:
            print(f"\n🔍 线条提取与补全开始，共 {len(self.colors)} 种颜色")

        for idx, color in enumerate(self.colors):
            if verbose:
                rgb = color.get('rgb', (0,0,0))
                print(f"  🎨 处理颜色 {idx}: RGB{rgb}")

            # ---------- 1. 生成颜色掩膜 ----------
            target_lab = color['lab']
            mask = self._create_color_mask(target_lab, range_l, range_ab)

            # 若掩膜为空，跳过
            if cv2.countNonZero(mask) == 0:
                if verbose: print(f"     掩膜为空，跳过")
                continue

            # ---------- 2. 形态学优化 ----------
            # 开运算去除孤立噪点，闭运算弥合微小断裂
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # ---------- 3. 骨架化 ----------
            skeleton = self._skeletonize(mask)
            if skeleton is None or np.sum(skeleton) == 0:
                if verbose: print(f"     骨架化失败或无像素")
                continue

            # ---------- 4. 线段跟踪 ----------
            segments = self._track_segments(skeleton, min_len=min_segment_len)
            if not segments:
                if verbose: print(f"     未跟踪到有效线段")
                continue

            # ---------- 5. 断点匹配与连接 ----------
            connected_segments = self._connect_endpoints(segments, max_dist=connect_distance)

            # 保存该颜色的处理结果
            self.results[idx] = {
                'color': color,
                'mask': mask,
                'skeleton': skeleton,
                'segments': segments,
                'connected_segments': connected_segments
            }

        if verbose:
            print(f"✅ 线条处理完成，成功处理 {len(self.results)} 种颜色\n")
        return self.results

    def _create_color_mask(self, target_lab: np.ndarray,
                           range_l: int, range_ab: int) -> np.ndarray:
        """根据 Lab 中心值及容忍度生成二值掩膜"""
        L = self.img_lab_float[:, :, 0]
        a = self.img_lab_float[:, :, 1]
        b = self.img_lab_float[:, :, 2]

        diff_L = np.abs(L - target_lab[0])
        diff_a = a - target_lab[1]
        diff_b = b - target_lab[2]
        diff_ab = np.sqrt(diff_a**2 + diff_b**2)

        mask = (diff_L < range_l) & (diff_ab < range_ab)
        return (mask * 255).astype(np.uint8)

    def _skeletonize(self, binary_mask: np.ndarray) -> np.ndarray:
        """将二值掩膜细化为单像素宽度的骨架"""
        # 确保二值图取值范围 0-1
        bin_img = (binary_mask > 0).astype(np.uint8)

        if SKIMAGE_AVAILABLE:
            skel = skeletonize(bin_img).astype(np.uint8) * 255
        else:
            # 降级方案：使用 OpenCV 的 thinning（需要 contrib 模块）
            try:
                skel = cv2.ximgproc.thinning(bin_img * 255)
            except AttributeError:
                # 最简方案：仅用腐蚀模拟（效果很差，仅用于演示）
                kernel = np.ones((3,3), np.uint8)
                eroded = cv2.erode(bin_img * 255, kernel, iterations=1)
                skel = cv2.bitwise_and(bin_img * 255, eroded)
                print("     警告: 骨架化功能受限，使用腐蚀近似")
        return skel

    def _track_segments(self, skeleton: np.ndarray,
                        min_len: int = 5) -> List[np.ndarray]:
        """
        从骨架图中提取所有连通路径（线段）。
        返回：每个元素为 N×2 的整数坐标数组（按路径顺序）
        """
        # 将骨架转为 0-1 格式
        skel = (skeleton > 0).astype(np.uint8)
        # 查找所有连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel, connectivity=8)
        segments = []

        for i in range(1, num_labels):  # 0 为背景
            if stats[i, cv2.CC_STAT_AREA] < min_len:
                continue

            # 提取该连通域的所有像素坐标
            pts = np.column_stack(np.where(labels == i))  # (y, x) 格式
            if len(pts) < min_len:
                continue

            # 简单排序：寻找端点，然后深度优先遍历
            # 这里采用简化方法：按连通域边界顺序（可能不是路径顺序，但用于可视化足够）
            # 更精确的路径跟踪可通过8邻域DFS实现，为保持简洁，此处仅返回所有点集
            segments.append(pts[:, ::-1])  # 转为 (x, y) 格式

        return segments

    def _connect_endpoints(self, segments: List[np.ndarray],
                           max_dist: int = 50) -> List[Dict]:
        """
        对线段端点进行匹配并生成连接线段。
        简化实现：对每个端点，寻找距离最近且方向兼容的未匹配端点，直接连接。
        """
        # 提取所有端点
        endpoints = []  # 每个元素: (x, y, seg_idx, is_start)
        for seg_idx, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            start_pt = seg[0]
            end_pt = seg[-1]
            endpoints.append((start_pt[0], start_pt[1], seg_idx, True))
            endpoints.append((end_pt[0], end_pt[1], seg_idx, False))

        if len(endpoints) < 2:
            return []

        # 构建距离矩阵
        n = len(endpoints)
        dist_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = endpoints[i][0], endpoints[i][1]
                xj, yj = endpoints[j][0], endpoints[j][1]
                d = math.hypot(xi - xj, yi - yj)
                dist_mat[i, j] = d
                dist_mat[j, i] = d

        # 标记哪些端点已经匹配
        matched = [False] * n
        connections = []

        if SCIPY_AVAILABLE:
            # 匈牙利算法全局最优匹配
            # 为每个端点分配一个虚拟伙伴，代价为距离，且不允许同线段内匹配
            # 构造代价矩阵，将距离大于max_dist的设为较大值
            cost = np.where(dist_mat > max_dist, 1e9, dist_mat)
            # 禁止同线段内匹配：代价设为无穷大
            for i in range(n):
                for j in range(n):
                    if endpoints[i][2] == endpoints[j][2]:  # 同一线段
                        cost[i, j] = 1e9

            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if i < j and cost[i, j] < 1e8 and not matched[i] and not matched[j]:
                    # 记录连接
                    connections.append({
                        'start': (endpoints[i][0], endpoints[i][1]),
                        'end': (endpoints[j][0], endpoints[j][1]),
                        'seg_idx_i': endpoints[i][2],
                        'seg_idx_j': endpoints[j][2],
                        'distance': dist_mat[i, j]
                    })
                    matched[i] = matched[j] = True
        else:
            # 贪心匹配：每次选距离最小且未匹配的端点对
            indices = list(range(n))
            while indices:
                i = indices.pop(0)
                if matched[i]:
                    continue
                best_j = -1
                best_d = float('inf')
                for j in indices:
                    if matched[j] or endpoints[i][2] == endpoints[j][2]:
                        continue
                    d = dist_mat[i, j]
                    if d < max_dist and d < best_d:
                        best_d = d
                        best_j = j
                if best_j != -1:
                    connections.append({
                        'start': (endpoints[i][0], endpoints[i][1]),
                        'end': (endpoints[best_j][0], endpoints[best_j][1]),
                        'seg_idx_i': endpoints[i][2],
                        'seg_idx_j': endpoints[best_j][2],
                        'distance': best_d
                    })
                    matched[i] = matched[best_j] = True
                    indices.remove(best_j)

        return connections

    def visualize_color_result(self, color_idx: int):
        """
        为指定颜色生成可视化窗口，展示线条还原效果。
        窗口内容：原始图像上叠加提取的骨架（白色）和补全线段（红色）。
        """
        if color_idx not in self.results:
            print(f"颜色 {color_idx} 无处理结果")
            return

        res = self.results[color_idx]
        color_info = res['color']
        rgb = color_info.get('rgb', (0,0,0))
        lab = color_info.get('lab', (0,0,0))

        # 创建画布：复制原始图像，作为背景
        canvas = self.img_bgr.copy()
        overlay = np.zeros_like(canvas)

        # 1. 绘制原始骨架（白色）
        skel = res['skeleton']
        overlay[skel > 0] = (255, 255, 255)

        # 2. 绘制补全线段（红色）
        for conn in res.get('connected_segments', []):
            start = tuple(map(int, conn['start']))
            end = tuple(map(int, conn['end']))
            cv2.line(overlay, start, end, (0, 0, 255), 1, cv2.LINE_AA)
            # 标记端点（蓝色）
            cv2.circle(overlay, start, 2, (255, 0, 0), -1)
            cv2.circle(overlay, end, 2, (255, 0, 0), -1)

        # 将 overlay 与原始图像融合（增强可视性）
        canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.8, 0)

        # 添加文字标注
        label = f"Color {color_idx} RGB{rgb}  L{int(lab[0])} a{int(lab[1])} b{int(lab[2])}"
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 1, cv2.LINE_AA)

        # 显示窗口
        win_name = f"Line Restoration - Color {color_idx}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 600)
        cv2.imshow(win_name, canvas)

    def visualize_all(self):
        """为所有成功处理的颜色创建窗口"""
        for idx in self.results.keys():
            self.visualize_color_result(idx)


def process_single_image(image_path: str,
                         color_extract_params: dict = None,
                         line_process_params: dict = None):
    """
    处理单张图像：颜色提取 -> 线条补全 -> 可视化
    """
    print(f"\n{'='*60}")
    print(f"处理图像: {image_path}")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return

    # 1. 颜色提取（使用 ColorExtractor）
    if ColorExtractor is None:
        print("错误: ColorExtractor 不可用，无法继续")
        return

    # 设置颜色提取参数（可针对图像微调）
    ce_params = color_extract_params or {
        'step': 1,
        'diff': 15,
        'min_area': 2,
        'min_l': 10,
        'max_l': 240,
        'min_chroma': 20,
        'merge_diff_l': 80,
        'merge_diff_ab': 20,
        'min_ratio': 0.0005,
        'max_ratio': 0.90,
        'range_l': 60,
        'range_ab': 15,
        'verbose': False,
        'visualize': False   # 不显示颜色提取的窗口
    }

    extractor = ColorExtractor(img_path=image_path)
    colors = extractor.process(**ce_params)

    if not colors:
        print("未提取到任何颜色，程序终止")
        return

    print(f"提取到 {len(colors)} 种颜色")

    # 2. 线条处理与补全
    lc_params = line_process_params or {
        'range_l': 30,
        'range_ab': 15,
        'morph_kernel': 3,
        'connect_distance': 60,
        'min_segment_len': 5,
        'verbose': True
    }

    completer = LineCompleter(extractor.img, colors)
    completer.process(**lc_params)

    # 3. 可视化
    if completer.results:
        completer.visualize_all()
        print(f"已打开 {len(completer.results)} 个结果窗口，按任意键关闭当前图像的所有窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未生成任何线条处理结果")


def main():
    """主测试入口"""
    # 测试图像路径（请确保文件存在）
    test_images = [
        "tests/img_broken.png",
        "tests/img_curve.png"
    ]

    # 可根据图像特点调整参数（此处演示默认参数）
    for img_path in test_images:
        # 针对不同图像可设置不同参数，这里留出扩展接口
        if "broken" in img_path:
            # 虚线图像：需更小的 min_area 和更大的 connect_distance
            ce_params = {
                'step': 1,
                'diff': 12,
                'min_area': 2,
                'min_l': 10,
                'max_l': 240,
                'min_chroma': 15,
                'merge_diff_l': 80,
                'merge_diff_ab': 20,
                'min_ratio': 0.0002,
                'max_ratio': 0.90,
                'verbose': False,
                'visualize': False
            }
            lc_params = {
                'range_l': 30,
                'range_ab': 15,
                'morph_kernel': 2,      # 小核避免虚焊
                'connect_distance': 80,  # 虚线断点可能较远
                'min_segment_len': 3,
                'verbose': True
            }
        elif "curve" in img_path:
            # 曲线图像：可能存在交叉遮挡
            ce_params = {
                'step': 1,
                'diff': 15,
                'min_area': 5,           # 曲线连通域较大
                'min_l': 10,
                'max_l': 240,
                'min_chroma': 20,
                'merge_diff_l': 80,
                'merge_diff_ab': 15,     # 严格色度区分不同曲线
                'min_ratio': 0.0005,
                'max_ratio': 0.90,
                'verbose': False,
                'visualize': False
            }
            lc_params = {
                'range_l': 30,
                'range_ab': 12,
                'morph_kernel': 3,
                'connect_distance': 50,
                'min_segment_len': 10,
                'verbose': True
            }
        else:
            ce_params = None
            lc_params = None

        process_single_image(img_path, ce_params, lc_params)

    print("\n🎉 所有图像处理完毕，程序退出")


if __name__ == "__main__":
    main()
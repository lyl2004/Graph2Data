"""
图表坐标轴提取工具 (Chart Axis Extractor) - 短线连接版

该类专用于从科学图表中自动提取X轴和Y轴的位置信息。
核心改进：连接短线成长线，直接筛选长直线作为坐标轴。
"""

import cv2
import numpy as np
import math
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import argparse


class AxisExtractor:
    """
    图表坐标轴提取器 (Chart Axis Extractor) - 短线连接版
    """

    def __init__(self, img_path: str):
        """
        初始化提取器实例。
        """
        self.img_path = img_path
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"错误: 找不到文件路径: {img_path}")

        self.img_raw = cv2.imread(img_path)
        if self.img_raw is None:
            raise ValueError("错误: 图像读取失败，文件可能已损坏或格式不受支持。")

        self.h, self.w = self.img_raw.shape[:2]

        # 默认配置参数
        self.cfg = {
            # 可视化颜色
            'color_x': (128, 0, 128),       # 深紫色
            'color_y': (0, 140, 255),       # 深橙色
            'color_dot_start': (0, 0, 255),  # 红色
            'color_dot_end': (255, 0, 0),    # 蓝色

            # 基础阈值
            'black_thresh': 100,

            # 霍夫变换参数
            'hough_min_len_ratio': 0.08,
            'hough_gap_ratio': 0.2,
            'hough_thresh_ratio': 0.06,
            
            # 短线连接参数
            'max_gap_for_connection': 30,      # 最大连接间隙（像素）
            'max_angle_diff_for_connection': 2.0,  # 最大角度差（度）
            'min_line_length_after_connect': 50,   # 连接后的最小线段长度
            
            # 长直线筛选参数
            'min_axis_length_ratio': 0.4,      # 坐标轴最小长度比例
            'angle_tolerance': 5.0,            # 角度容差
            
            # 显示设置
            'display_max_height': 600,
            'screen_max_width': 1800,
        }

        # 预处理方案开关
        self.use_min_channel = False
        self.use_adaptive = False
        self.use_morph = False
        self.use_cuda = False
        self.no_plot = False

        # 调试缓存
        self.debug_cache: Dict[str, np.ndarray] = {}
        self.extraction_result: Optional[Dict[str, Any]] = None

    def process(self,
                # 预处理方案开关
                use_min_channel: bool = False,
                use_adaptive: bool = False,
                use_morph: bool = False,
                use_cuda: bool = False,
                no_plot: bool = False,
                # 霍夫变换核心参数
                hough_thresh_ratio: float = 0.06,
                hough_min_len_ratio: float = 0.08,
                hough_gap_ratio: float = 0.2,
                black_thresh: int = 100,
                # 短线连接参数
                max_gap_for_connection: int = 30,
                max_angle_diff_for_connection: float = 2.0,
                min_line_length_after_connect: int = 50,
                # 长直线筛选参数
                min_axis_length_ratio: float = 0.4,
                angle_tolerance: float = 5.0,
                # 可视化参数
                show_steps: List[str] = None,
                verbose: bool = True) -> Optional[Dict[str, Any]]:
        """
        执行核心坐标轴提取流程。
        """
        # 更新配置
        self.use_min_channel = use_min_channel
        self.use_adaptive = use_adaptive
        self.use_morph = use_morph
        self.use_cuda = use_cuda
        self.no_plot = no_plot

        self.cfg.update({
            'hough_thresh_ratio': hough_thresh_ratio,
            'hough_min_len_ratio': hough_min_len_ratio,
            'hough_gap_ratio': hough_gap_ratio,
            'black_thresh': black_thresh,
            'max_gap_for_connection': max_gap_for_connection,
            'max_angle_diff_for_connection': max_angle_diff_for_connection,
            'min_line_length_after_connect': min_line_length_after_connect,
            'min_axis_length_ratio': min_axis_length_ratio,
            'angle_tolerance': angle_tolerance,
        })

        if verbose:
            print(f"🚀 [启动] 正在分析图像: {self.img_path}")
            print(f"📊 [图像] 尺寸: {self.w}x{self.h} 像素")

        # 记录开始时间
        start_time = time.time()

        # ==========================================
        # 阶段 1: 预处理与二值化
        # ==========================================
        if verbose:
            print(f"🛠️  [阶段 1] 执行图像预处理...")

        img_rot, angle, binary = self._preprocess_and_rotate(self.img_raw)
        self.debug_cache['1_binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # ==========================================
        # 阶段 2: 自适应参数计算
        # ==========================================
        long_side = max(self.h, self.w)

        min_line_len = max(20, int(long_side * self.cfg['hough_min_len_ratio']))
        max_line_gap = max(5, int(min_line_len * self.cfg['hough_gap_ratio']))
        hough_thresh = max(30, int(long_side * self.cfg['hough_thresh_ratio']))

        if verbose:
            print(f"⚙️  [阶段 2] 计算自适应参数...")
            print(f"   → 最小线长: {min_line_len} 像素")
            print(f"   → 最大间隔: {max_line_gap} 像素")
            print(f"   → 霍夫阈值: {hough_thresh}")

        # ==========================================
        # 阶段 3: 霍夫变换线段检测
        # ==========================================
        if verbose:
            print(f"📐 [阶段 3] 执行霍夫变换线段检测...")

        lines = cv2.HoughLinesP(binary, 1, np.pi / 180,
                                threshold=hough_thresh,
                                minLineLength=min_line_len,
                                maxLineGap=max_line_gap)

        # 可视化霍夫检测结果
        hough_vis = img_rot.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if verbose:
                print(f"   → 检测到 {len(lines)} 条候选线段")
        else:
            if verbose:
                print("   ⚠️  未检测到任何线段")
            self.extraction_result = {
                'origin': None,
                'x_end': None,
                'y_end': None,
                'success': False,
                'error': 'No lines detected',
                'processing_time': time.time() - start_time
            }
            return self.extraction_result

        self.debug_cache['2_hough'] = hough_vis

        # ==========================================
        # 阶段 4: 短线连接和长直线筛选（重构部分）
        # ==========================================
        if verbose:
            print(f"🎯 [阶段 4] 执行短线连接和长直线筛选...")

        # 新的短线连接算法
        connected_lines = self._connect_short_lines(lines, binary.shape)
        
        if verbose:
            print(f"   → 连接后得到 {len(connected_lines)} 条线段")
        
        # 从连接后的线段中筛选坐标轴
        x_axis, y_axis = self._find_axes_from_connected_lines(connected_lines, (self.h, self.w))

        if x_axis and y_axis:
            rough_y = x_axis['pos']
            rough_x = y_axis['pos']

            # 提取轴线范围限制
            x_lines_xs = [p for l in x_axis['lines'] for p in (l[0], l[2])]
            y_lines_ys = [p for l in y_axis['lines'] for p in (l[1], l[3])]

            x_range_limit = (min(x_lines_xs), max(x_lines_xs))
            y_range_limit = (min(y_lines_ys), max(y_lines_ys))

            if verbose:
                print(f"   → 检测到候选轴线: X轴(水平)在 y={rough_y:.1f}, Y轴(垂直)在 x={rough_x:.1f}")
                print(f"   → X轴线段数: {len(x_axis['lines'])}")
                print(f"   → Y轴线段数: {len(y_axis['lines'])}")

            # ==========================================
            # 阶段 5: 轴线位置精修
            # ==========================================
            if verbose:
                print(f"🔍 [阶段 5] 执行轴线位置精修...")

            refine_range = 8
            final_y = self._refine_position(binary, rough_y, axis=0,
                                            search_range=refine_range,
                                            limit_range=x_range_limit)
            final_x = self._refine_position(binary, rough_x, axis=1,
                                            search_range=refine_range,
                                            limit_range=y_range_limit)

            # 计算关键点坐标
            origin = (int(final_x), int(final_y))
            x_end = (int(max(x_lines_xs)), int(final_y))
            y_end = (int(final_x), int(min(y_lines_ys)))

            # ==========================================
            # 阶段 6: 结果绘制与存储
            # ==========================================
            res_img = self._draw_final_result(img_rot, origin, x_end, y_end)
            self.debug_cache['3_result'] = res_img

            # 构建结果字典
            self.extraction_result = {
                'origin': origin,
                'x_end': x_end,
                'y_end': y_end,
                'x_axis': {
                    'position': final_y,
                    'range': x_range_limit,
                    'segment_count': len(x_axis['lines'])
                },
                'y_axis': {
                    'position': final_x,
                    'range': y_range_limit,
                    'segment_count': len(y_axis['lines'])
                },
                'image_size': (self.w, self.h),
                'success': True,
                'processing_time': time.time() - start_time
            }

            if verbose:
                print(f"✅ [完成] 坐标轴提取成功!")
                print(f"   → 原点坐标: {origin}")
                print(f"   → X轴终点: {x_end}")
                print(f"   → Y轴终点: {y_end}")
                print(f"   → 处理时间: {time.time() - start_time:.3f}秒")

        else:
            if verbose:
                print("❌ [失败] 未找到有效的坐标轴线对")
                if not x_axis:
                    print("   → 未找到X轴（水平线）")
                if not y_axis:
                    print("   → 未找到Y轴（垂直线）")
            
            self.extraction_result = {
                'origin': None,
                'x_end': None,
                'y_end': None,
                'success': False,
                'error': 'No valid axis pair found',
                'processing_time': time.time() - start_time
            }
            self.debug_cache['3_result'] = img_rot

        # ==========================================
        # 阶段 7: 可视化展示
        # ==========================================
        if not self.no_plot:
            if show_steps is None:
                show_steps = ['binary', 'hough', 'result']
            self._visualize_results(show_steps)

        return self.extraction_result

    def _preprocess_and_rotate(self, img: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        集成方案 1, 2, 3 的增强预处理流程。
        """
        # --- 方案 1: 最小通道提取 (针对浅色线条) ---
        if self.use_min_channel:
            gray = np.min(img, axis=2)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- 方案 2: 自适应阈值 (针对光照不均或线条极细) ---
        if self.use_adaptive:
            binary = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           15, 10)
        else:
            if self.use_min_channel:
                _, binary = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(gray, self.cfg['black_thresh'],
                                          255, cv2.THRESH_BINARY_INV)

        # --- 方案 3: 形态学运算 (连接断线) ---
        if self.use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 返回原始图像（当前版本不旋转）、旋转角度（0）和二值图像
        return img, 0.0, binary

    def _connect_short_lines(self, lines: np.ndarray, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        连接短线成长线的核心算法
        
        算法步骤:
        1. 提取所有线段并计算角度、长度、方向
        2. 按角度分组（水平线和垂直线分开处理）
        3. 在每个组内，将共线且端点靠近的线段连接起来
        4. 避免连接文字区域等复杂区域的密集短线
        
        Args:
            lines: 霍夫变换检测到的线段
            image_shape: 图像尺寸 (高度, 宽度)
            
        Returns:
            连接后的线段列表
        """
        h, w = image_shape
        connected_lines = []
        
        # 提取线段信息
        line_info_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            
            # 计算角度（标准化到0-180度）
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 180
            
            length = math.hypot(dx, dy)
            
            line_info_list.append({
                'coords': (x1, y1, x2, y2),
                'angle': angle,
                'length': length,
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'used': False  # 标记是否已连接
            })
        
        # 按角度分组：水平线（角度接近0或180度）和垂直线（角度接近90度）
        horizontal_lines = []
        vertical_lines = []
        
        for line_info in line_info_list:
            angle = line_info['angle']
            
            # 水平线检测（角度接近0或180度）
            if angle < self.cfg['angle_tolerance'] or angle > 180 - self.cfg['angle_tolerance']:
                horizontal_lines.append(line_info)
            
            # 垂直线检测（角度接近90度）
            elif 90 - self.cfg['angle_tolerance'] < angle < 90 + self.cfg['angle_tolerance']:
                vertical_lines.append(line_info)
        
        # 分别处理水平线和垂直线
        connected_horizontal = self._connect_lines_in_group(horizontal_lines, 'horizontal', w)
        connected_vertical = self._connect_lines_in_group(vertical_lines, 'vertical', h)
        
        # 合并所有连接后的线段
        connected_lines.extend(connected_horizontal)
        connected_lines.extend(connected_vertical)
        
        return connected_lines
    
    def _connect_lines_in_group(self, lines: List[Dict], line_type: str, 
                               reference_length: int) -> List[Dict]:
        """
        连接同一组内的线段
        
        Args:
            lines: 同一方向的线段列表
            line_type: 线段类型 ('horizontal' 或 'vertical')
            reference_length: 参考长度（用于计算距离阈值）
            
        Returns:
            连接后的线段列表
        """
        if not lines:
            return []
        
        # 按位置排序
        if line_type == 'horizontal':
            # 水平线按y坐标排序
            lines.sort(key=lambda x: x['center'][1])
        else:
            # 垂直线按x坐标排序
            lines.sort(key=lambda x: x['center'][0])
        
        connected = []
        
        for i in range(len(lines)):
            if lines[i]['used']:
                continue
                
            current_line = lines[i]
            connected_lines = [current_line]
            current_line['used'] = True
            
            # 尝试连接其他线段
            for j in range(i + 1, len(lines)):
                if lines[j]['used']:
                    continue
                
                # 检查是否应该连接
                if self._should_connect_lines(current_line, lines[j], line_type, reference_length):
                    connected_lines.append(lines[j])
                    lines[j]['used'] = True
                    
                    # 更新当前线为连接后的线
                    current_line = self._merge_lines(connected_lines, line_type)
            
            # 创建连接后的线段
            merged_line = self._create_merged_line(connected_lines, line_type)
            
            # 检查连接后的线段是否足够长
            if merged_line['length'] >= self.cfg['min_line_length_after_connect']:
                connected.append(merged_line)
        
        return connected
    
    def _should_connect_lines(self, line1: Dict, line2: Dict, 
                             line_type: str, reference_length: int) -> bool:
        """
        判断两条线段是否应该连接
        
        Args:
            line1: 第一条线段
            line2: 第二条线段
            line_type: 线段类型
            reference_length: 参考长度
            
        Returns:
            是否应该连接
        """
        # 检查角度是否接近
        angle_diff = abs(line1['angle'] - line2['angle'])
        if angle_diff > self.cfg['max_angle_diff_for_connection']:
            return False
        
        # 提取线段端点
        x11, y11, x12, y12 = line1['coords']
        x21, y21, x22, y22 = line2['coords']
        
        # 计算端点之间的距离
        if line_type == 'horizontal':
            # 对于水平线，主要检查y坐标是否接近，x坐标是否有重叠或接近
            avg_y1 = (y11 + y12) / 2
            avg_y2 = (y21 + y22) / 2
            
            # y坐标必须非常接近
            if abs(avg_y1 - avg_y2) > self.cfg['max_gap_for_connection'] / 2:
                return False
            
            # 检查x坐标是否接近或有重叠
            min_x1, max_x1 = min(x11, x12), max(x11, x12)
            min_x2, max_x2 = min(x21, x22), max(x21, x22)
            
            # 检查是否有重叠
            has_overlap = not (max_x1 < min_x2 - self.cfg['max_gap_for_connection'] or 
                             min_x1 > max_x2 + self.cfg['max_gap_for_connection'])
            
            # 检查间隙是否在允许范围内
            gap = min(abs(max_x1 - min_x2), abs(min_x1 - max_x2))
            
            return has_overlap or gap <= self.cfg['max_gap_for_connection']
            
        else:  # vertical
            # 对于垂直线，主要检查x坐标是否接近，y坐标是否有重叠或接近
            avg_x1 = (x11 + x12) / 2
            avg_x2 = (x21 + x22) / 2
            
            # x坐标必须非常接近
            if abs(avg_x1 - avg_x2) > self.cfg['max_gap_for_connection'] / 2:
                return False
            
            # 检查y坐标是否接近或有重叠
            min_y1, max_y1 = min(y11, y12), max(y11, y12)
            min_y2, max_y2 = min(y21, y22), max(y21, y22)
            
            # 检查是否有重叠
            has_overlap = not (max_y1 < min_y2 - self.cfg['max_gap_for_connection'] or 
                             min_y1 > max_y2 + self.cfg['max_gap_for_connection'])
            
            # 检查间隙是否在允许范围内
            gap = min(abs(max_y1 - min_y2), abs(min_y1 - max_y2))
            
            return has_overlap or gap <= self.cfg['max_gap_for_connection']
    
    def _merge_lines(self, lines: List[Dict], line_type: str) -> Dict:
        """
        合并多条线段为一条线段
        
        Args:
            lines: 要合并的线段列表
            line_type: 线段类型
            
        Returns:
            合并后的线段信息
        """
        if not lines:
            return None
        
        # 提取所有端点
        all_coords = [line['coords'] for line in lines]
        
        if line_type == 'horizontal':
            # 对于水平线，找到最左和最右的x坐标，取平均y坐标
            all_x = [x for coords in all_coords for x in (coords[0], coords[2])]
            all_y = [y for coords in all_coords for y in (coords[1], coords[3])]
            
            min_x, max_x = min(all_x), max(all_x)
            avg_y = np.mean(all_y)
            
            # 创建新的线段（水平）
            new_coords = (min_x, avg_y, max_x, avg_y)
            
        else:  # vertical
            # 对于垂直线，找到最上和最下的y坐标，取平均x坐标
            all_x = [x for coords in all_coords for x in (coords[0], coords[2])]
            all_y = [y for coords in all_coords for y in (coords[1], coords[3])]
            
            min_y, max_y = min(all_y), max(all_y)
            avg_x = np.mean(all_x)
            
            # 创建新的线段（垂直）
            new_coords = (avg_x, min_y, avg_x, max_y)
        
        # 计算新线段的属性
        x1, y1, x2, y2 = new_coords
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 180
        length = math.hypot(dx, dy)
        
        return {
            'coords': new_coords,
            'angle': angle,
            'length': length,
            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
            'used': True
        }
    
    def _create_merged_line(self, lines: List[Dict], line_type: str) -> Dict:
        """
        创建合并线段的最终表示
        
        Args:
            lines: 原始线段列表
            line_type: 线段类型
            
        Returns:
            合并线段的表示
        """
        merged = self._merge_lines(lines, line_type)
        
        if not merged:
            return None
        
        # 提取原始线段坐标（用于后续处理）
        original_coords = [line['coords'] for line in lines]
        
        return {
            'coords': merged['coords'],
            'angle': merged['angle'],
            'length': merged['length'],
            'center': merged['center'],
            'original_lines': original_coords,
            'line_count': len(lines)
        }
    
    def _find_axes_from_connected_lines(self, connected_lines: List[Dict], 
                                       shape: Tuple[int, int]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        从连接后的线段中筛选坐标轴
        
        Args:
            connected_lines: 连接后的线段列表
            shape: 图像尺寸 (高度, 宽度)
            
        Returns:
            最佳的水平轴和垂直轴
        """
        if not connected_lines:
            return None, None
        
        h, w = shape
        
        # 参数
        min_h_length = w * self.cfg['min_axis_length_ratio']
        min_v_length = h * self.cfg['min_axis_length_ratio']
        
        # 分离水平线和垂直线
        horizontal_candidates = []
        vertical_candidates = []
        
        for line in connected_lines:
            angle = line['angle']
            
            # 水平线筛选
            if angle < self.cfg['angle_tolerance'] or angle > 180 - self.cfg['angle_tolerance']:
                if line['length'] >= min_h_length:
                    x1, y1, x2, y2 = line['coords']
                    horizontal_candidates.append({
                        'coords': (x1, y1, x2, y2),
                        'length': line['length'],
                        'avg_y': (y1 + y2) / 2,
                        'center': line['center']
                    })
            
            # 垂直线筛选
            elif 90 - self.cfg['angle_tolerance'] < angle < 90 + self.cfg['angle_tolerance']:
                if line['length'] >= min_v_length:
                    x1, y1, x2, y2 = line['coords']
                    vertical_candidates.append({
                        'coords': (x1, y1, x2, y2),
                        'length': line['length'],
                        'avg_x': (x1 + x2) / 2,
                        'center': line['center']
                    })
        
        # 打印候选线段信息
        print(f"   → 长水平线候选: {len(horizontal_candidates)}条")
        print(f"   → 长垂直线候选: {len(vertical_candidates)}条")
        
        # 选择最长的水平线作为X轴
        best_vertical = None
        if vertical_candidates:
            # 1. 首先过滤掉太短的线 (例如小于最大长度的 70%)
            max_len = max([v['length'] for v in vertical_candidates])
            long_candidates = [v for v in vertical_candidates if v['length'] > max_len * 0.7]
            
            # 2. 在长线候选中，按 x 坐标排序 (优先选左边的)
            # x['center'][0] 是线段中心的 x 坐标
            long_candidates.sort(key=lambda x: x['center'][0]) 
            
            best_v = long_candidates[0] # 取最左边的长线
            
            best_vertical = {
                'pos': best_v['avg_x'],
                'lines': [best_v['coords']],
                'len': best_v['length']
            }
        
        best_horizontal = None
        if horizontal_candidates:
            # 1. 过滤短线
            max_len = max([h_line['length'] for h_line in horizontal_candidates])
            long_candidates = [h_line for h_line in horizontal_candidates if h_line['length'] > max_len * 0.7]
            
            # 2. 在长线候选中，按 y 坐标降序排序 (优先选下边的，y值越大越靠下)
            long_candidates.sort(key=lambda x: x['center'][1], reverse=True)
            
            best_h = long_candidates[0] # 取最下边的长线
            
            best_horizontal = {
                'pos': best_h['avg_y'],
                'lines': [best_h['coords']],
                'len': best_h['length']
            }
        # 验证轴线配对合理性
        if best_horizontal and best_vertical:
            if not self._validate_axis_pair_simple(best_horizontal, best_vertical, w, h):
                print("   ⚠️  轴线配对不合理，尝试其他候选...")
                # 尝试第二长的候选
                if len(horizontal_candidates) > 1 and len(vertical_candidates) > 1:
                    second_h = horizontal_candidates[1]
                    second_v = vertical_candidates[1]
                    
                    new_horizontal = {
                        'pos': second_h['avg_y'],
                        'lines': [second_h['coords']],
                        'len': second_h['length']
                    }
                    new_vertical = {
                        'pos': second_v['avg_x'],
                        'lines': [second_v['coords']],
                        'len': second_v['length']
                    }
                    
                    if self._validate_axis_pair_simple(new_horizontal, new_vertical, w, h):
                        return new_horizontal, new_vertical
        
        return best_horizontal, best_vertical
    
    def _find_axes_with_lower_threshold(self, connected_lines: List[Dict], 
                                       h: int, w: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        使用较低的长度阈值寻找轴线
        
        Args:
            connected_lines: 连接后的线段列表
            h: 图像高度
            w: 图像宽度
            
        Returns:
            轴线对
        """
        # 降低长度阈值
        min_h_length = w * self.cfg['min_axis_length_ratio'] * 0.7  # 降低到70%
        min_v_length = h * self.cfg['min_axis_length_ratio'] * 0.7
        
        horizontal_candidates = []
        vertical_candidates = []
        
        for line in connected_lines:
            angle = line['angle']
            
            # 水平线筛选
            if angle < self.cfg['angle_tolerance'] or angle > 180 - self.cfg['angle_tolerance']:
                if line['length'] >= min_h_length:
                    x1, y1, x2, y2 = line['coords']
                    horizontal_candidates.append({
                        'coords': (x1, y1, x2, y2),
                        'length': line['length'],
                        'avg_y': (y1 + y2) / 2
                    })
            
            # 垂直线筛选
            elif 90 - self.cfg['angle_tolerance'] < angle < 90 + self.cfg['angle_tolerance']:
                if line['length'] >= min_v_length:
                    x1, y1, x2, y2 = line['coords']
                    vertical_candidates.append({
                        'coords': (x1, y1, x2, y2),
                        'length': line['length'],
                        'avg_x': (x1 + x2) / 2
                    })
        
        # 选择最长的候选
        best_horizontal = None
        if horizontal_candidates:
            horizontal_candidates.sort(key=lambda x: x['length'], reverse=True)
            best_h = horizontal_candidates[0]
            best_horizontal = {
                'pos': best_h['avg_y'],
                'lines': [best_h['coords']],
                'len': best_h['length']
            }
        
        best_vertical = None
        if vertical_candidates:
            vertical_candidates.sort(key=lambda x: x['length'], reverse=True)
            best_v = vertical_candidates[0]
            best_vertical = {
                'pos': best_v['avg_x'],
                'lines': [best_v['coords']],
                'len': best_v['length']
            }
        
        return best_horizontal, best_vertical
    
    def _validate_axis_pair_simple(self, horizontal_axis: Dict, vertical_axis: Dict,
                                  img_width: int, img_height: int) -> bool:
        """
        简单的轴线配对验证
        
        Args:
            horizontal_axis: 水平轴线
            vertical_axis: 垂直轴线
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            配对是否合理
        """
        h_pos = horizontal_axis['pos']
        v_pos = vertical_axis['pos']
        
        # 基本合理性检查
        # 1. 轴线位置应该在图像范围内
        if h_pos < 0 or h_pos > img_height or v_pos < 0 or v_pos > img_width:
            return False
        
        # 2. 轴线长度应该足够
        if horizontal_axis['len'] < img_width * 0.3 or vertical_axis['len'] < img_height * 0.3:
            return False
        
        # 3. 对于大多数图表，水平线应该在垂直线的下方
        # 获取垂直线的底部（最大y坐标）
        v_lines = vertical_axis['lines']
        v_y_coords = [y for line in v_lines for y in (line[1], line[3])]
        v_bottom = max(v_y_coords) if v_y_coords else 0
        
        # 允许一定的容差，但水平线不应该在垂直线上方太远
        if h_pos < v_bottom - img_height * 0.2:  # 水平线在垂直线上方20%以上
            return False
        
        return True

    def _refine_position(self, binary: np.ndarray, rough_pos: float,
                         axis: int, search_range: int,
                         limit_range: Tuple[int, int]) -> int:
        """
        通过像素密度分析精修轴线位置。
        """
        h, w = binary.shape
        start = int(max(0, rough_pos - search_range))
        end_pos = int(min(binary.shape[1 - axis], rough_pos + search_range))
        best_pos = int(rough_pos)
        max_density = -1

        limit_min, limit_max = map(int, limit_range)
        limit_min = max(0, limit_min)

        for i in range(start, end_pos):
            count = 0
            if axis == 0:  # 水平线：统计指定x范围内的y=i线上的像素
                l_max = min(w, limit_max)
                if l_max > limit_min:
                    count = np.count_nonzero(binary[i, limit_min:l_max])
            else:  # 垂直线：统计指定y范围内的x=i线上的像素
                l_max = min(h, limit_max)
                if l_max > limit_min:
                    count = np.count_nonzero(binary[limit_min:l_max, i])

            if count > max_density:
                max_density = count
                best_pos = i

        return best_pos

    def _draw_final_result(self, img: np.ndarray,
                           origin: Tuple[int, int],
                           x_end: Tuple[int, int],
                           y_end: Tuple[int, int]) -> np.ndarray:
        """
        绘制最终的坐标轴提取结果。
        """
        vis = img.copy()

        # 绘制轴线
        cv2.line(vis, origin, x_end, self.cfg['color_x'], 2)
        cv2.line(vis, origin, y_end, self.cfg['color_y'], 2)

        # 绘制辅助线（坐标框）
        top_right = (x_end[0], y_end[1])
        cv2.line(vis, y_end, top_right, (200, 200, 200), 1)
        cv2.line(vis, x_end, top_right, (200, 200, 200), 1)

        # 绘制关键点
        cv2.circle(vis, origin, 5, self.cfg['color_dot_start'], -1)  # 原点（红）
        cv2.circle(vis, x_end, 5, self.cfg['color_dot_start'], -1)  # X轴终点（红）
        cv2.circle(vis, y_end, 5, self.cfg['color_dot_end'], -1)    # Y轴终点（蓝）

        return vis

    def _visualize_results(self, show_steps: List[str]):
        """
        生成可视化分析窗口。
        """
        print(f"🖼️  [UI] 正在生成可视化界面...")

        # 收集需要显示的图像
        valid_imgs = []
        step_map = {
            'binary': '1_binary',
            'hough': '2_hough',
            'result': '3_result'
        }

        for step_key, cache_key in step_map.items():
            if step_key in show_steps and cache_key in self.debug_cache:
                img = self.debug_cache[cache_key].copy()
                # 添加步骤标签
                cv2.putText(img, step_key, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                valid_imgs.append(img)

        if not valid_imgs:
            return

        # 调整图像尺寸
        target_h = self.cfg['display_max_height']
        resized_imgs = []
        for img in valid_imgs:
            h, w = img.shape[:2]
            if h > 0:
                scale = target_h / h
                resized_imgs.append(cv2.resize(img, (int(w * scale), target_h)))

        # 水平拼接
        final_view = np.hstack(resized_imgs)

        # 限制最大宽度
        if final_view.shape[1] > self.cfg['screen_max_width']:
            scale = self.cfg['screen_max_width'] / final_view.shape[1]
            final_view = cv2.resize(final_view, (0, 0), fx=scale, fy=scale)

        # 显示窗口
        cv2.imshow("Axis Extractor - Debug View", final_view)
        print("🏁 [等待] 请按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图表坐标轴提取工具 (Chart Axis Extractor) - 短线连接版")

    # 基础输入
    parser.add_argument("--img_path", type=str, required=True,
                        help="输入图像的路径 (支持 .jpg, .png)。")

    # 预处理方案开关
    parser.add_argument("--use_min_channel", action="store_true",
                        help="启用最小通道提取方案：针对浅色线条（如青/黄）丢失的问题。")
    parser.add_argument("--use_adaptive", action="store_true",
                        help="启用自适应阈值方案：针对光照不均或背景灰暗的图片。")
    parser.add_argument("--use_morph", action="store_true",
                        help="启用形态学闭运算方案：针对虚线或断断续续的线条。")

    # 霍夫变换核心参数
    parser.add_argument("--hough_thresh_ratio", type=float, default=0.06,
                        help="霍夫投票阈值比例 (默认 0.06)。越大越严格，越小越灵敏。")
    parser.add_argument("--hough_min_len_ratio", type=float, default=0.08,
                        help="最小线长比例 (默认 0.08)。短于此长度的线段会被忽略。")
    parser.add_argument("--hough_gap_ratio", type=float, default=0.2,
                        help="最大断裂间隔比例 (默认 0.2)。允许线段中间断开的最大距离。")
    parser.add_argument("--black_thresh", type=int, default=100,
                        help="全局二值化阈值 (默认 100, 范围0-255)。仅在未开启自适应阈值时生效。")

    # 短线连接参数
    parser.add_argument("--max_gap_for_connection", type=int, default=30,
                        help="最大连接间隙 (默认 30 像素)。")
    parser.add_argument("--max_angle_diff_for_connection", type=float, default=2.0,
                        help="最大角度差 (默认 2.0 度)。")
    parser.add_argument("--min_line_length_after_connect", type=int, default=50,
                        help="连接后的最小线段长度 (默认 50 像素)。")

    # 长直线筛选参数
    parser.add_argument("--min_axis_length_ratio", type=float, default=0.4,
                        help="坐标轴最小长度比例 (默认 0.4)。")
    parser.add_argument("--angle_tolerance", type=float, default=5.0,
                        help="角度容差 (默认 5.0 度)。")

    # 系统与显示参数
    parser.add_argument("--use_cuda", action="store_true",
                        help="启用CUDA加速 (需要OpenCV编译CUDA支持)。")
    parser.add_argument("--no_plot", action="store_true",
                        help="静默模式：不弹出显示窗口，适合命令行批处理或服务器运行。")
    parser.add_argument("--silent", action="store_true",
                        help="静默模式：不输出控制台日志。")

    args = parser.parse_args()

    # 执行主程序
    try:
        extractor = AxisExtractor(args.img_path)

        result = extractor.process(
            use_min_channel=args.use_min_channel,
            use_adaptive=args.use_adaptive,
            use_morph=args.use_morph,
            use_cuda=args.use_cuda,
            no_plot=args.no_plot,
            hough_thresh_ratio=args.hough_thresh_ratio,
            hough_min_len_ratio=args.hough_min_len_ratio,
            hough_gap_ratio=args.hough_gap_ratio,
            black_thresh=args.black_thresh,
            max_gap_for_connection=args.max_gap_for_connection,
            max_angle_diff_for_connection=args.max_angle_diff_for_connection,
            min_line_length_after_connect=args.min_line_length_after_connect,
            min_axis_length_ratio=args.min_axis_length_ratio,
            angle_tolerance=args.angle_tolerance,
            show_steps=['binary', 'hough', 'result'],
            verbose=not args.silent
        )

        if result and result['success']:
            print(f"\n=== 坐标轴提取结果 ===")
            print(f"原点坐标: {result['origin']}")
            print(f"X轴终点: {result['x_end']}")
            print(f"Y轴终点: {result['y_end']}")
            print(f"X轴位置: y={result['x_axis']['position']}")
            print(f"Y轴位置: x={result['y_axis']['position']}")
            if 'processing_time' in result:
                print(f"处理时间: {result['processing_time']:.3f}秒")
        else:
            print(f"坐标轴提取失败: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ 程序执行出错: {str(e)}")
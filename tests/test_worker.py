"""
图表分析工作流脚本 (Chart Analysis Workflow) - 独立窗口高清晰版
- 内存OCR，无临时文件
- 调试阶段精细化控制（1=仅坐标轴，2=坐标轴+OCR，3=坐标轴+OCR+颜色提取）
- 全seaborn可视化，独立窗口，超大字体，高亮线条
- 强制使用本地中文字体
"""

import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ========== 强制中文字体加载（无回退，不存在即崩溃） ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)          # 项目根目录
font_path = os.path.join(project_root, 'assets', 'simsun.ttc')

if not os.path.exists(font_path):
    raise FileNotFoundError(
        f"\n❌ 严重错误：中文字体文件不存在 - {font_path}\n"
        f"请将 simsun.ttc 放置在 {os.path.join(project_root, 'assets')} 目录下。"
    )

try:
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 字体加载成功: {font_path} (Family: {font_name})")
except Exception as e:
    raise RuntimeError(f"❌ 字体加载失败: {e}") from e

# ========== seaborn 全局样式设置（确保字体不被覆盖） ==========
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['axes.unicode_minus'] = False

# ========== 导入项目内部模块 ==========
sys.path.append(project_root)
from test_pre import AxisExtractor
from rapidocr_onnxruntime import RapidOCR
from test_colors import ColorExtractor


class ChartWorkflow:
    """图表分析工作流 - 内存版（独立窗口，高亮显示）"""

    def __init__(self, img_path: str):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self._ocr_engine = None
        if self.img is None:
            raise ValueError("错误: 图像读取失败")

        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img.shape[:2]

        # 存储中间结果
        self.axis_result: Optional[Dict[str, Any]] = None
        self.region_polygons: Dict[int, List[Tuple[int, int]]] = {}
        self.ocr_results: Dict[int, List[Dict[str, Any]]] = {}
        self.color_results: Optional[List[Dict[str, Any]]] = None
        self.global_ocr_index = 0
        self.region_boundary_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        # 颜色定义（seaborn风格 + 高亮自定义）
        self.colors = {
            'region_border': sns.color_palette("Set2")[2],
            'region_text': sns.color_palette("Set2")[3],
            'axis_line': sns.color_palette("Set2")[0],
            'origin_point': sns.color_palette("Set2")[1],
            'corner_line': sns.color_palette("Set2")[4],
            'ocr_box': 'darkorange',
        }
        # 高亮区域边界颜色（按顺序：上、右、下、左）
        self.border_colors = ['cyan', 'hotpink', 'limegreen', 'yellow']

    # -------------------- 阶段1：坐标轴检测 --------------------
    def run_axis_detection_silent(self, **kwargs) -> bool:
        """静默执行坐标轴检测（不弹出OpenCV窗口）"""
        try:
            extractor = AxisExtractor(self.img_path)
            self.axis_result = extractor.process(
                use_min_channel=kwargs.get('use_min_channel', False),
                use_adaptive=kwargs.get('use_adaptive', False),
                use_morph=kwargs.get('use_morph', False),
                no_plot=True,
                verbose=False,
                show_steps=[],
                hough_thresh_ratio=0.04,
                hough_min_len_ratio=0.05,
            )
            success = self.axis_result is not None and self.axis_result.get('success', False)
            if success:
                print("✅ 坐标轴检测成功")
                origin = self.axis_result['origin']
                x_end = self.axis_result['x_end']
                y_end = self.axis_result['y_end']
                print(f"   原点: {origin}")
                print(f"   X轴终点: {x_end}")
                print(f"   Y轴终点: {y_end}")
            else:
                print("❌ 坐标轴检测失败")
            return success
        except Exception as e:
            print(f"❌ 坐标轴检测错误: {e}")
            return False

    # -------------------- 阶段2：区域定义 --------------------
    def define_regions_with_overlap(self) -> bool:
        """定义区域，允许重叠区域的重复记录"""
        if not self.axis_result or not self.axis_result.get('success', False):
            return False
        try:
            origin = self.axis_result['origin']
            x_end = self.axis_result['x_end']
            y_end = self.axis_result['y_end']

            x0, y0 = origin[0], origin[1]
            x1, y0_x = x_end[0], x_end[1]
            x0_y, y1 = y_end[0], y_end[1]

            rect_left = min(x0, x0_y)
            rect_right = max(x0, x1)
            rect_top = min(y0, y1)
            rect_bottom = max(y0, y0_x)

            top_left = (rect_left, rect_top)
            top_right = (rect_right, rect_top)
            bottom_left = (rect_left, rect_bottom)
            bottom_right = (rect_right, rect_bottom)

            self.region_boundary_lines = [
                (top_left, top_right),
                (top_right, bottom_right),
                (bottom_right, bottom_left),
                (bottom_left, top_left)
            ]

            print("\n" + "="*60)
            print("调试1: 区域边界线坐标")
            print("="*60)
            line_names = ["上边界", "右边界", "下边界", "左边界"]
            for i, (start, end) in enumerate(self.region_boundary_lines):
                print(f"   {line_names[i]}: ({start[0]}, {start[1]}) -> ({end[0]}, {end[1]})")

            region0 = [top_left, top_right, bottom_right, bottom_left]
            region1 = [(0, 0), (0, rect_top), top_left, bottom_left, (0, rect_bottom), (0, self.h-1)]
            region2 = [(0, self.h-1), (0, rect_bottom), bottom_left, bottom_right, (self.w-1, rect_bottom), (self.w-1, self.h-1)]
            region3 = [(self.w-1, 0), (self.w-1, rect_top), top_right, bottom_right, (self.w-1, rect_bottom), (self.w-1, self.h-1)]
            region4 = [(0, 0), (0, rect_top), top_left, top_right, (self.w-1, rect_top), (self.w-1, 0)]

            self.region_polygons = {0: region0, 1: region1, 2: region2, 3: region3, 4: region4}

            self.corner_lines = [
                (top_left, (0, rect_top)),
                (top_left, (rect_left, 0)),
                (top_right, (self.w-1, rect_top)),
                (top_right, (rect_right, 0)),
                (bottom_left, (0, rect_bottom)),
                (bottom_left, (rect_left, self.h-1)),
                (bottom_right, (self.w-1, rect_bottom)),
                (bottom_right, (rect_right, self.h-1)),
            ]

            self.rect_boundary = {'left': rect_left, 'right': rect_right, 'top': rect_top, 'bottom': rect_bottom}

            print(f"\n📐 坐标轴矩形边界:")
            print(f"   左上角: {top_left}")
            print(f"   右上角: {top_right}")
            print(f"   左下角: {bottom_left}")
            print(f"   右下角: {bottom_right}")
            print(f"   宽度: {rect_right - rect_left} 像素")
            print(f"   高度: {rect_bottom - rect_top} 像素")
            return True
        except Exception as e:
            print(f"❌ 区域定义错误: {e}")
            return False

    # -------------------- 区域图像提取（内存操作）--------------------
    def extract_region_image_with_offset(self, region_id: int) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        if region_id not in self.region_polygons:
            return None, (0, 0)
        vertices = self.region_polygons[region_id]
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        region_img = cv2.bitwise_and(self.img, self.img, mask=mask)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None, (0, 0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        cropped_img = region_img[y_min:y_max+1, x_min:x_max+1]
        if cropped_img.size == 0:
            return None, (0, 0)

        # 图像增强（仅用于OCR区域）
        if region_id != 0:
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                cropped_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            elif mean_brightness > 200:
                cropped_img = cv2.convertScaleAbs(cropped_img, alpha=0.7, beta=0)

        return cropped_img, (x_min, y_min)

    # -------------------- 阶段3：内存OCR --------------------
    def run_ocr_on_all_regions_silent(self, **kwargs) -> Dict[int, List[Dict[str, Any]]]:
        print("\n" + "="*60)
        print("开始OCR文本提取（内存模式）")
        print("="*60)

        self.ocr_results = {}
        self.global_ocr_index = 0

        ocr_params = {
            'db_thresh': kwargs.get('db_thresh', 0.1),
            'box_thresh': kwargs.get('box_thresh', 0.3),
            'unclip_ratio': kwargs.get('unclip_ratio', 2.0),
            'rec_thresh': kwargs.get('rec_thresh', 0.1),
        }

        if self._ocr_engine is None:
            print("   ⚙️ 初始化OCR引擎...")
            self._ocr_engine = RapidOCR(
                det_db_thresh=ocr_params['db_thresh'],
                det_db_unclip_ratio=ocr_params['unclip_ratio'],
                det_db_box_thresh=ocr_params['box_thresh'],
                rec_thresh=ocr_params['rec_thresh'],
                det_use_cuda=kwargs.get('use_cuda', False)
            )

        region_names = {0: "有效曲线区域", 1: "Y轴定义与刻度", 2: "X轴定义与刻度", 3: "右侧区域", 4: "上方区域"}

        for region_id in range(5):
            print(f"\n🔍 处理区域 {region_id} ({region_names.get(region_id, '未知')})...")
            region_img, (x_offset, y_offset) = self.extract_region_image_with_offset(region_id)
            if region_img is None:
                print(f"   ⚠️  区域图像为空，跳过")
                self.ocr_results[region_id] = []
                continue

            h_img, w_img = region_img.shape[:2]
            print(f"   区域图像尺寸: {w_img}x{h_img}, 偏移量: x={x_offset}, y={y_offset}")
            if w_img < 10 or h_img < 10:
                print(f"   ⚠️  图像尺寸过小，跳过")
                self.ocr_results[region_id] = []
                continue

            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
            processed_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            try:
                result_group, _ = self._ocr_engine(processed_img, use_det=True, use_cls=False, use_rec=True)
                global_results = []
                if result_group:
                    for item in result_group:
                        box, text, confidence = item
                        if confidence < ocr_params['rec_thresh']:
                            continue
                        box_array = np.array(box, dtype=np.float32)
                        global_box = (box_array + [x_offset, y_offset]).tolist()
                        center = np.mean(box_array, axis=0) + [x_offset, y_offset]
                        result_entry = {
                            'global_index': self.global_ocr_index,
                            'region_id': region_id,
                            'x_offset': x_offset,
                            'y_offset': y_offset,
                            'text': text,
                            'confidence': confidence,
                            'box_boundary': box,
                            'box_boundary_global': global_box,
                            'center_global': center.tolist()
                        }
                        global_results.append(result_entry)
                        self.global_ocr_index += 1
                print(f"   ✅ OCR完成: 检测到 {len(global_results)} 个文本")
                self.ocr_results[region_id] = global_results
            except Exception as e:
                print(f"   ❌ OCR处理错误: {e}")
                self.ocr_results[region_id] = []

        return self.ocr_results

    def display_ocr_results_in_terminal(self):
        print("\n" + "="*60)
        print("调试2: OCR识别结果（按区域分组）")
        print("="*60)
        total_count = 0
        region_names = {0: "有效曲线区域", 1: "Y轴定义与刻度", 2: "X轴定义与刻度", 3: "右侧区域", 4: "上方区域"}
        for region_id in range(5):
            results = self.ocr_results.get(region_id, [])
            if results:
                print(f"\n区域 {region_id} ({region_names.get(region_id, '未知')}):")
                print("  " + "-" * 50)
                for result in results:
                    idx = result.get('global_index', 0)
                    text = result.get('text', '').strip()
                    confidence = result.get('confidence', 0)
                    center = result.get('center_global', (0, 0))
                    coord_info = f"中心坐标: ({center[0]:.1f}, {center[1]:.1f})"
                    if text:
                        print(f"  [{idx:3d}] 置信度: {confidence:.3f} | {coord_info} | 内容: {text}")
                        total_count += 1
                valid = len([r for r in results if r.get('text', '').strip()])
                print(f"  本区域有效文本: {valid} 个")
            else:
                print(f"\n区域 {region_id} ({region_names.get(region_id, '未知')}): 未检测到文本")
        print(f"\n所有区域总计有效文本: {total_count} 个")
        print("="*60)

    # -------------------- 阶段3扩展：区域0颜色提取 --------------------
    def run_color_extraction_on_region0(self, color_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        print("\n" + "="*60)
        print("🎨 开始颜色提取（区域0 - 有效曲线区域）")
        print("="*60)
        if 0 not in self.region_polygons:
            print("❌ 区域0未定义，无法提取颜色")
            return []
        region_img, offset = self.extract_region_image_with_offset(0)
        if region_img is None:
            print("❌ 区域0图像为空，跳过颜色提取")
            return []

        h_img, w_img = region_img.shape[:2]
        print(f"   区域0图像尺寸: {w_img}x{h_img}, 偏移量: x={offset[0]}, y={offset[1]}")

        default_params = {
            'step': 1, 'diff': 15, 'min_area': 2,
            'min_l': 10, 'max_l': 240, 'min_chroma': 10,
            'merge_diff_l': 80, 'merge_diff_ab': 20,
            'min_ratio': 0.001, 'max_ratio': 0.90,
            'range_l': 60, 'range_ab': 15,
            'verbose': False, 'visualize': False,
        }
        if color_params:
            default_params.update(color_params)

        try:
            extractor = ColorExtractor(img_array=region_img)
            colors = extractor.process(**default_params)
            self.color_results = colors
            print(f"✅ 颜色提取完成，检测到 {len(colors)} 种颜色")
            return colors
        except Exception as e:
            print(f"❌ 颜色提取错误: {e}")
            return []

    def display_color_results_in_terminal(self):
        print("\n" + "="*60)
        print("调试3: 区域0颜色提取结果")
        print("="*60)
        if not self.color_results:
            print("未检测到颜色")
            return
        print(f"{'ID':<5} | {'RGB':<20} | {'Lab':<25} | {'占比':<10} | {'面积':<10}")
        print("-"*60)
        for idx, c in enumerate(self.color_results):
            rgb = c['rgb']
            lab = c['lab']
            ratio = c['ratio']
            area = c['area']
            print(f"{idx:<5} | ({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d}) | "
                  f"({lab[0]:3.0f},{lab[1]:3.0f},{lab[2]:3.0f}) | {ratio:8.4%} | {area:<10}")
        print("="*60)

    # -------------------- 调试图像生成（独立窗口，超大字体，高亮线条）--------------------
    def create_debug_image_1(self) -> np.ndarray:
        """调试图像1：坐标轴+区域划分（独立窗口用）"""
        sns.set_theme(style="whitegrid", rc={'font.sans-serif': [font_name], 'axes.unicode_minus': False})
        fig, ax = plt.subplots(figsize=(24, 16), dpi=150)

        ax.imshow(self.img_rgb)
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)

        # ---- 坐标轴（线宽5，深蓝色） ----
        if self.axis_result and self.axis_result.get('success', False):
            origin = self.axis_result['origin']
            x_end = self.axis_result['x_end']
            y_end = self.axis_result['y_end']
            ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]],
                   color=self.colors['axis_line'], linewidth=5)
            ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]],
                   color=self.colors['axis_line'], linewidth=5)
            ax.scatter([origin[0], x_end[0], y_end[0]],
                      [origin[1], x_end[1], y_end[1]],
                      c=[self.colors['origin_point'], self.colors['origin_point'], self.colors['axis_line']],
                      s=120, zorder=5)

        # ---- 角点连线（线宽3，紫色虚线） ----
        if hasattr(self, 'corner_lines'):
            for start, end in self.corner_lines:
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       color=self.colors['corner_line'], linewidth=3, linestyle='--', alpha=0.8)

        # ---- 区域边界：四条边分别使用高亮色，线宽5 ----
        if hasattr(self, 'region_boundary_lines') and len(self.region_boundary_lines) == 4:
            for i, (start, end) in enumerate(self.region_boundary_lines):
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       color=self.border_colors[i], linewidth=5, alpha=0.9)

        # ---- 绘制所有5个区域的轮廓（使用统一颜色，但透明度低一些，作为辅助）----
        for region_id, vertices in self.region_polygons.items():
            x_coords = [v[0] for v in vertices] + [vertices[0][0]]
            y_coords = [v[1] for v in vertices] + [vertices[0][1]]
            ax.plot(x_coords, y_coords, color='gray', linewidth=2, alpha=0.4, linestyle=':')
            # 区域编号（大号加粗）
            center_x = np.mean(x_coords[:-1])
            center_y = np.mean(y_coords[:-1])
            ax.text(center_x, center_y, f'区域{region_id}',
                   fontsize=18, fontweight='bold', color=self.colors['region_text'],
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor=self.colors['region_text'], alpha=0.9))

        # 原点标注
        if self.axis_result and self.axis_result.get('success', False):
            origin = self.axis_result['origin']
            ax.text(origin[0], origin[1], '原点', fontsize=14, color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))

        ax.set_title('调试图像1: 坐标轴与区域划分', fontsize=28, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return img_array

    def create_debug_image_2(self) -> np.ndarray:
        """调试图像2：坐标轴+区域划分+OCR识别结果（继承图像1所有绘制）"""
        sns.set_theme(style="whitegrid", rc={'font.sans-serif': [font_name], 'axes.unicode_minus': False})
        fig, ax = plt.subplots(figsize=(24, 16), dpi=150)

        ax.imshow(self.img_rgb)
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)

        # ===== 继承图像1的所有绘制元素 =====
        # 坐标轴（线宽5，深蓝色）
        if self.axis_result and self.axis_result.get('success', False):
            origin = self.axis_result['origin']
            x_end = self.axis_result['x_end']
            y_end = self.axis_result['y_end']
            ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]],
                   color=self.colors['axis_line'], linewidth=5)
            ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]],
                   color=self.colors['axis_line'], linewidth=5)
            ax.scatter([origin[0], x_end[0], y_end[0]],
                      [origin[1], x_end[1], y_end[1]],
                      c=[self.colors['origin_point'], self.colors['origin_point'], self.colors['axis_line']],
                      s=120, zorder=5)

        # 角点连线（线宽3，紫色虚线）
        if hasattr(self, 'corner_lines'):
            for start, end in self.corner_lines:
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       color=self.colors['corner_line'], linewidth=3, linestyle='--', alpha=0.8)

        # 区域边界：四条高亮边（线宽5）
        if hasattr(self, 'region_boundary_lines') and len(self.region_boundary_lines) == 4:
            for i, (start, end) in enumerate(self.region_boundary_lines):
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       color=self.border_colors[i], linewidth=5, alpha=0.9)

        # 所有5个区域的轮廓（浅灰虚线）及区域编号（大号加粗）
        for region_id, vertices in self.region_polygons.items():
            x_coords = [v[0] for v in vertices] + [vertices[0][0]]
            y_coords = [v[1] for v in vertices] + [vertices[0][1]]
            ax.plot(x_coords, y_coords, color='gray', linewidth=2, alpha=0.4, linestyle=':')
            center_x = np.mean(x_coords[:-1])
            center_y = np.mean(y_coords[:-1])
            ax.text(center_x, center_y, f'区域{region_id}',
                   fontsize=22, fontweight='bold', color=self.colors['region_text'],
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                            edgecolor=self.colors['region_text'], alpha=0.9))

        # 原点标注
        if self.axis_result and self.axis_result.get('success', False):
            origin = self.axis_result['origin']
            ax.text(origin[0], origin[1], '原点', fontsize=16, color='red',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red'))

        # ===== 叠加 OCR 检测框（深橙色，线宽5） =====
        box_color = 'darkorange'
        for region_id in range(5):
            results = self.ocr_results.get(region_id, [])
            for result in results:
                if 'box_boundary_global' not in result:
                    continue
                box_points = result['box_boundary_global']
                if not box_points or len(box_points) < 4:
                    continue
                x_coords = [p[0] for p in box_points] + [box_points[0][0]]
                y_coords = [p[1] for p in box_points] + [box_points[0][1]]
                ax.plot(x_coords, y_coords, color=box_color, linewidth=5, alpha=0.9)

                idx = result.get('global_index', 0)
                confidence = result.get('confidence', 0)
                conf_percent = confidence * 100
                label_text = f"{idx};{conf_percent:.1f}%"

                min_x = min(x_coords[:-1])
                max_x = max(x_coords[:-1])
                min_y = min(y_coords[:-1])
                max_y = max(y_coords[:-1])
                text_x = (min_x + max_x) / 2
                text_y = min_y - 15

                ax.text(text_x, text_y, label_text,
                       fontsize=18, fontweight='bold', color=box_color,
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=box_color, linewidth=2.5, alpha=0.95))

        total_texts = sum(len(results) for results in self.ocr_results.values())
        ax.set_title(f'调试图像2: 坐标轴+区域+OCR (总计{total_texts}个文本)',
                    fontsize=32, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return img_array

    def create_debug_image_3(self) -> np.ndarray:
        """调试图像3：区域0颜色色卡（纯色块，右侧规范格式文字，全加粗）"""
        sns.set_theme(style="whitegrid", rc={'font.sans-serif': [font_name], 'axes.unicode_minus': False})

        if not self.color_results:
            fig, ax = plt.subplots(figsize=(24, 12), dpi=150)
            ax.text(0.5, 0.5, "区域0未检测到颜色", ha='center', va='center',
                   fontsize=32, fontweight='bold')
            ax.axis('off')
        else:
            num = len(self.color_results)
            fig, axes = plt.subplots(num, 1, figsize=(28, 5 * num), dpi=150)
            if num == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                c = self.color_results[i]
                rgb = c['rgb']
                lab = c['lab']
                ratio = c['ratio']

                # ---------- 左侧：纯色块（无文字）----------
                rect_width = 0.6      # 占轴宽度60%
                rect_height = 0.5     # 占轴高度50%
                rect_x = 0.05
                rect_y = 0.25
                ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                          transform=ax.transAxes,
                                          color=np.array(rgb)/255,
                                          linewidth=0))
                # ---------- 右侧：规范格式文字（全加粗黑色）----------
                text_x = rect_x + rect_width + 0.1
                text_y_start = 0.8

                # 第一行：颜色序号 + Lab + RGB
                line1 = f'颜色{i} ； lab（{lab[0]:.0f}，{lab[1]:.0f}，{lab[2]:.0f}）；RGB （{rgb[0]}，{rgb[1]}，{rgb[2]}）'
                ax.text(text_x, text_y_start, line1,
                       transform=ax.transAxes,
                       fontsize=20, fontweight='bold', color='black',
                       ha='left', va='top')

                # 第二行：面积占比
                line2 = f'面积占比：{ratio:.2%}'
                ax.text(text_x, text_y_start - 0.25, line2,
                       transform=ax.transAxes,
                       fontsize=20, fontweight='bold', color='black',
                       ha='left', va='top')

                # 设置子图范围，关闭坐标轴
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('auto')
                ax.axis('off')

            plt.suptitle("调试图像3: 区域0颜色提取色卡", fontsize=32, fontweight='bold', y=0.98)

        plt.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return img_array
    # -------------------- 独立窗口显示 --------------------
    def show_debug_images(self, stage: int):
        """根据阶段分别弹出独立窗口，每个窗口最大化显示"""
        if stage not in [1, 2, 3]:
            return

        # 窗口1
        if stage >= 1:
            print("\n生成调试图像1...")
            img1 = self.create_debug_image_1()
            fig1 = plt.figure(figsize=(24, 16), dpi=120)
            plt.imshow(img1)
            plt.title("调试图像1: 坐标轴与区域划分", fontsize=24, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()

        # 窗口2
        if stage >= 2:
            print("生成调试图像2...")
            img2 = self.create_debug_image_2()
            fig2 = plt.figure(figsize=(24, 16), dpi=120)
            plt.imshow(img2)
            plt.title("调试图像2: OCR识别结果", fontsize=24, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()

        # 窗口3
        if stage >= 3:
            print("生成调试图像3...")
            img3 = self.create_debug_image_3()
            fig3 = plt.figure(figsize=(24, 16), dpi=120)
            plt.imshow(img3)
            plt.title("调试图像3: 区域0颜色色卡", fontsize=24, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()

        plt.show()  # 一次性显示所有窗口

    # -------------------- 完整分析流程 --------------------
    def run_full_analysis(self,
                         axis_params: Dict[str, Any] = None,
                         ocr_params: Dict[str, Any] = None,
                         color_params: Dict[str, Any] = None,
                         stop_at_stage: int = 2) -> Dict[str, Any]:
        if axis_params is None:
            axis_params = {}
        if ocr_params is None:
            ocr_params = {}
        if color_params is None:
            color_params = {}

        result = {
            'image_info': {'path': self.img_path, 'size': (self.w, self.h)},
            'axis_detection': None, 'regions': {}, 'ocr_results': {},
            'color_results': None, 'success': False, 'stage_reached': 0
        }

        print("="*60)
        print("开始图表分析工作流")
        print("="*60)

        # 阶段1
        print("\n📐 步骤1: 坐标轴检测")
        print("-"*40)
        axis_success = self.run_axis_detection_silent(**axis_params)
        if not axis_success:
            print("❌ 坐标轴检测失败，停止分析")
            return result
        result['axis_detection'] = self.axis_result
        result['stage_reached'] = 1
        if stop_at_stage <= 1:
            print("\n⏹️  已达到指定终止阶段 (1)，分析结束")
            result['success'] = True
            return result

        # 阶段2
        print("\n🗺️  步骤2: 区域定义")
        print("-"*40)
        region_success = self.define_regions_with_overlap()
        if not region_success:
            print("❌ 区域定义失败，停止分析")
            return result
        for region_id, vertices in self.region_polygons.items():
            result['regions'][region_id] = {'vertices': vertices, 'vertex_count': len(vertices)}

        print("\n🔍 步骤3: OCR文本提取（内存模式）")
        print("-"*40)
        ocr_results = self.run_ocr_on_all_regions_silent(**ocr_params)
        result['ocr_results'] = ocr_results
        self.display_ocr_results_in_terminal()
        result['stage_reached'] = 2
        if stop_at_stage <= 2:
            print("\n⏹️  已达到指定终止阶段 (2)，分析结束")
            result['success'] = True
            return result

        # 阶段3
        print("\n🎨 步骤4: 区域0颜色提取（内存模式）")
        print("-"*40)
        color_results = self.run_color_extraction_on_region0(color_params)
        result['color_results'] = color_results
        self.display_color_results_in_terminal()
        result['stage_reached'] = 3
        result['success'] = True

        print("\n✅ 完整分析完成! (阶段3)")
        print("="*60)
        return result


def main():
    parser = argparse.ArgumentParser(description="图表分析工作流 - 独立窗口高清晰版")
    parser.add_argument("--img_path", type=str, default="tests/test1.png", help="输入图像路径")
    parser.add_argument("--use_min_channel", action="store_true", help="启用最小通道提取")
    parser.add_argument("--use_adaptive", action="store_true", help="启用自适应阈值")
    parser.add_argument("--use_morph", action="store_true", help="启用形态学运算")
    parser.add_argument("--db_thresh", type=float, default=0.1, help="OCR检测阈值")
    parser.add_argument("--rec_thresh", type=float, default=0.1, help="OCR识别阈值")
    parser.add_argument("--color_min_ratio", type=float, default=0.001, help="颜色最小占比阈值")
    parser.add_argument("--color_merge_diff_ab", type=int, default=20, help="颜色合并色度容差")
    parser.add_argument("--debugger", type=int, choices=[1, 2, 3], default=None,
                       help="调试阶段: 1=仅坐标轴+调试图1, 2=完整流程+调试图1&2, 3=完整流程+调试图1&2&3")

    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"❌ 错误: 文件不存在 {args.img_path}")
        return

    try:
        workflow = ChartWorkflow(args.img_path)
        axis_params = {
            'use_min_channel': args.use_min_channel,
            'use_adaptive': args.use_adaptive,
            'use_morph': args.use_morph,
        }
        ocr_params = {
            'db_thresh': args.db_thresh,
            'rec_thresh': args.rec_thresh,
        }
        color_params = {
            'min_ratio': args.color_min_ratio,
            'merge_diff_ab': args.color_merge_diff_ab,
        }
        stop_at_stage = 2 if args.debugger is None else args.debugger

        result = workflow.run_full_analysis(
            axis_params=axis_params,
            ocr_params=ocr_params,
            color_params=color_params,
            stop_at_stage=stop_at_stage
        )

        if args.debugger is not None:
            workflow.show_debug_images(args.debugger)

    except Exception as e:
        print(f"❌ 工作流执行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
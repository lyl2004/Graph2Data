"""
图表文本OCR提取工具 (Chart Text OCR Extractor)

该类专用于从科学图表中提取文本信息，包括坐标轴标签、图例文本等。
针对性解决了以下技术难点：
1. 小文本检测：通过图像填充和缩放增强小文本识别率。
2. 合并数字拆分：通过空格检测和几何分析拆分粘连的数字。
3. 非文本过滤：通过角度和形状分析过滤几何符号和垂直文本。
"""

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import os
from typing import Dict, List, Tuple, Optional, Any
import argparse
import re
import math


class OCRExtractor:
    """
    图表文本OCR提取器 (Chart Text OCR Extractor)

    该类专用于从科学图表中提取文本信息，包括坐标轴标签、图例文本等。
    通过专门的后处理算法，优化科学图表中的文本识别效果。
    """

    def __init__(self, img_path: str):
        """
        初始化提取器实例。

        Args:
            img_path (str): 图像文件的系统路径。

        Raises:
            FileNotFoundError: 如果路径不存在。
            ValueError: 如果OpenCV无法解码图像文件。
        """
        self.img_path = img_path
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"错误: 找不到文件路径: {img_path}")

        self.img_origin = cv2.imread(img_path)
        if self.img_origin is None:
            raise ValueError("错误: 图像读取失败，文件可能已损坏或格式不受支持。")

        self.h, self.w = self.img_origin.shape[:2]
        self.total_pixels = self.h * self.w

        # 默认配置参数
        self.cfg = {
            # OCR核心参数
            'db_thresh': 0.3,
            'box_thresh': 0.5,
            'unclip_ratio': 1.2,
            'rec_thresh': 0.6,
            
            # 后处理参数
            'max_text_angle': 30.0,
            
            # 可视化参数
            'color_boundary': (0, 140, 255),  # 橙色
            'color_text_bg': (255, 255, 255),  # 白色
            
            # 预处理参数
            'padding_size': 50,
            'scale_factor': 2.0,
        }

        # 处理开关
        self.enable_split_numbers = True
        self.filter_geometric_symbols = True
        self.remove_vertical_text = True
        self.use_cuda = False
        self.no_plot = False

        # 结果存储
        self.ocr_results: List[Dict[str, Any]] = []

    def process(self,
                # OCR核心参数
                db_thresh: float = 0.3,
                box_thresh: float = 0.5,
                unclip_ratio: float = 1.2,
                rec_thresh: float = 0.6,
                # 后处理开关
                enable_split_numbers: bool = True,
                filter_geometric_symbols: bool = True,
                remove_vertical_text: bool = True,
                max_text_angle: float = 30.0,
                # 系统参数
                use_cuda: bool = False,
                no_plot: bool = False,
                # 可视化参数
                verbose: bool = True) -> List[Dict[str, Any]]:
        """
        执行核心OCR提取流程。

        流程包括：图像预处理 → OCR检测识别 → 后处理优化 → 结果输出。

        Args:
            db_thresh (float): 文本检测阈值，越低检测越敏感。
            box_thresh (float): 文本框阈值，控制文本框的生成。
            unclip_ratio (float): 文本框扩展比例，用于处理紧凑文本。
            rec_thresh (float): 文本识别置信度阈值，低于此值的识别结果将被过滤。

            enable_split_numbers (bool): 启用合并数字拆分，针对粘连的数字序列。
            filter_geometric_symbols (bool): 过滤几何符号（如短横线、图例标记）。
            remove_vertical_text (bool): 过滤垂直文本（通常不是坐标轴标签）。
            max_text_angle (float): 最大文本角度，用于过滤非水平文本。

            use_cuda (bool): 启用CUDA加速。
            no_plot (bool): 静默模式，不弹出显示窗口。

            verbose (bool): 是否打印控制台日志。

        Returns:
            List[Dict[str, Any]]: 提取到的文本信息列表，包含内容、位置、置信度等。
        """
        # 更新配置
        self.cfg.update({
            'db_thresh': db_thresh,
            'box_thresh': box_thresh,
            'unclip_ratio': unclip_ratio,
            'rec_thresh': rec_thresh,
            'max_text_angle': max_text_angle,
        })

        self.enable_split_numbers = enable_split_numbers
        self.filter_geometric_symbols = filter_geometric_symbols
        self.remove_vertical_text = remove_vertical_text
        self.use_cuda = use_cuda
        self.no_plot = no_plot

        if verbose:
            print(f"🚀 [启动] 正在分析图像: {self.img_path}")
            print(f"📊 [图像] 尺寸: {self.w}x{self.h} 像素")

        # ==========================================
        # 阶段 1: 图像预处理
        # 目的: 增强小文本检测效果
        # ==========================================
        if verbose:
            print(f"🛠️  [阶段 1] 执行图像预处理...")

        img_enhanced, pad_size, scale_factor = self._preprocess_image(self.img_origin)
        if verbose:
            print(f"   → 填充尺寸: {pad_size} 像素")
            print(f"   → 缩放因子: {scale_factor:.1f}x")
            print(f"   → 处理后尺寸: {img_enhanced.shape[1]}x{img_enhanced.shape[0]}")

        # ==========================================
        # 阶段 2: OCR检测与识别
        # 目的: 提取图像中的所有文本区域和内容
        # ==========================================
        if verbose:
            print(f"🔍 [阶段 2] 执行OCR检测与识别...")

        ocr_engine = RapidOCR(
            det_db_thresh=self.cfg['db_thresh'],
            det_db_unclip_ratio=self.cfg['unclip_ratio'],
            det_limit_side_len=max(img_enhanced.shape[:2]),
            det_limit_type='max',
            det_use_cuda=self.use_cuda
        )

        # 执行OCR
        result_group, _ = ocr_engine(img_enhanced, use_det=True, use_cls=False, use_rec=True)

        # 初始候选结果
        candidates = []
        if result_group:
            for item in result_group:
                box, text, confidence = item
                
                # 过滤低置信度结果
                if confidence < self.cfg['rec_thresh']:
                    continue
                
                # 还原坐标到原始图像空间
                original_box = self._restore_box_coords(box, pad_size, scale_factor)
                
                # 过滤越界框
                if np.min(original_box) < -20:
                    continue
                
                candidates.append({
                    "text": text,
                    "confidence": confidence,
                    "box_boundary": original_box.tolist(),
                    "box_center": np.mean(original_box, axis=0).tolist()
                })

            if verbose:
                print(f"   → 初步检测到 {len(candidates)} 个文本区域")

        # ==========================================
        # 阶段 3: 后处理优化
        # 目的: 针对图表文本特点优化识别结果
        # ==========================================
        if verbose:
            print(f"🎯 [阶段 3] 执行后处理优化...")

        # 3.1 合并数字拆分
        if self.enable_split_numbers:
            candidates = self._process_merged_numbers(candidates)
            if verbose:
                print(f"   → 合并数字拆分完成")

        # 3.2 文本过滤
        candidates = self._filter_candidates(candidates)
        if verbose:
            print(f"   → 过滤后剩余 {len(candidates)} 个文本区域")

        # 3.3 结果排序（按行排列）
        candidates.sort(key=lambda x: (int(x['box_center'][1] / 15), x['box_center'][0]))
        self.ocr_results = candidates

        # ==========================================
        # 阶段 4: 结果输出与可视化
        # ==========================================
        if verbose:
            self._print_results_summary()

        if not self.no_plot:
            self._visualize_results()

        return self.ocr_results

    def _preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        图像预处理：填充和缩放以增强小文本检测。

        Args:
            img (np.ndarray): 原始BGR图像。

        Returns:
            Tuple[np.ndarray, int, float]: 处理后的图像、填充尺寸、缩放因子。
        """
        padding_size = self.cfg['padding_size']
        scale_factor = self.cfg['scale_factor']

        # 添加白色边框，避免边缘文本被截断
        img_padded = cv2.copyMakeBorder(img, padding_size, padding_size,
                                        padding_size, padding_size,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # 放大图像以增强小文本
        h, w = img_padded.shape[:2]
        img_resized = cv2.resize(img_padded,
                                 (int(w * scale_factor), int(h * scale_factor)),
                                 interpolation=cv2.INTER_CUBIC)

        return img_resized, padding_size, scale_factor

    def _restore_box_coords(self, box: List[List[float]],
                            pad_size: int, scale_factor: float) -> np.ndarray:
        """
        将检测框坐标还原到原始图像空间。

        Args:
            box (List[List[float]]): 检测框坐标，四个点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]。
            pad_size (int): 预处理时的填充尺寸。
            scale_factor (float): 预处理时的缩放因子。

        Returns:
            np.ndarray: 还原后的检测框坐标。
        """
        box_array = np.array(box, dtype=np.float32)
        
        # 1. 反向缩放
        box_array /= scale_factor
        
        # 2. 去除填充偏移
        box_array -= pad_size
        
        return box_array

    def _calculate_box_angle(self, box: np.ndarray) -> float:
        """
        计算文本框的倾斜角度。

        Args:
            box (np.ndarray): 检测框坐标，形状为 (4, 2)。

        Returns:
            float: 文本框的角度（度）。
        """
        p1, p2 = box[0], box[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dy, dx))

    def _process_merged_numbers(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理合并的数字：通过空格检测拆分粘连的数字序列。

        Args:
            candidates (List[Dict[str, Any]]): 原始候选结果。

        Returns:
            List[Dict[str, Any]]: 拆分后的候选结果。
        """
        new_candidates = []
        pattern = re.compile(r'^[\d\.\-]+\s+[\d\.\-]+')

        for item in candidates:
            text = item['text'].strip()
            box = np.array(item['box_boundary'])

            # 检查是否包含空格且符合数字序列模式
            if ' ' in text and pattern.match(text):
                parts = text.split()
                parts = [p for p in parts if p.strip()]  # 过滤空字符串
                n_parts = len(parts)

                if n_parts > 1:
                    # 几何切割：假设字符等宽分布
                    xs = box[:, 0]
                    min_x, max_x = np.min(xs), np.max(xs)
                    total_width = max_x - min_x
                    step = total_width / n_parts

                    ys = box[:, 1]
                    min_y, max_y = np.min(ys), np.max(ys)

                    for i, part in enumerate(parts):
                        # 构造子框
                        sub_min_x = min_x + i * step
                        sub_max_x = min_x + (i + 1) * step

                        sub_box = np.array([
                            [sub_min_x, min_y],
                            [sub_max_x, min_y],
                            [sub_max_x, max_y],
                            [sub_min_x, max_y]
                        ], dtype=np.float32)

                        center = np.mean(sub_box, axis=0)

                        new_entry = item.copy()
                        new_entry['text'] = part
                        new_entry['box_boundary'] = sub_box.tolist()
                        new_entry['box_center'] = center.tolist()
                        new_candidates.append(new_entry)
                else:
                    new_candidates.append(item)
            else:
                new_candidates.append(item)

        return new_candidates

    def _filter_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤候选结果：移除几何符号、垂直文本等非目标文本。

        Args:
            candidates (List[Dict[str, Any]]): 原始候选结果。

        Returns:
            List[Dict[str, Any]]: 过滤后的候选结果。
        """
        filtered_candidates = []

        for item in candidates:
            text = item['text'].strip()
            box = np.array(item['box_boundary'])

            # 1. 角度过滤：移除垂直文本
            if self.remove_vertical_text:
                angle = self._calculate_box_angle(box)
                if abs(angle) > self.cfg['max_text_angle']:
                    continue

            # 2. 几何符号过滤：移除图例中的短横线等
            if self.filter_geometric_symbols:
                xs, ys = box[:, 0], box[:, 1]
                width = np.max(xs) - np.min(xs)
                height = np.max(ys) - np.min(ys)
                aspect_ratio = height / (width + 1e-6)

                # 过滤正方形或接近正方形的短横线（通常是图例符号）
                if text in ['-', '—', '--', '_'] and aspect_ratio > 0.6:
                    continue

            filtered_candidates.append(item)

        return filtered_candidates

    def _print_results_summary(self):
        """打印结果摘要到控制台。"""
        print("\n" + "=" * 50)
        print(f"OCR 提取结果: {os.path.basename(self.img_path)}")
        print(f"总计检测: {len(self.ocr_results)} 个文本区域")
        print("=" * 50)
        print(f"{'ID':<5} | {'内容':<20} | {'置信度':<6} | {'中心坐标':<15}")
        print("-" * 50)

        for idx, item in enumerate(self.ocr_results):
            clean_text = item['text'].strip().replace('\n', ' ')
            center_x, center_y = int(item['box_center'][0]), int(item['box_center'][1])
            print(f"{idx:<5} | {clean_text:<20} | {item['confidence']:.2f}  | ({center_x}, {center_y})")

        print("=" * 50 + "\n")

    def _visualize_results(self):
        """
        生成可视化分析窗口。
        包含: 原始图像、检测框标注结果。
        """
        print(f"🖼️  [UI] 正在生成可视化界面...")

        img_vis = self.img_origin.copy()

        # 绘制检测框和编号
        for idx, item in enumerate(self.ocr_results):
            box = np.array(item['box_boundary'], dtype=np.int32)
            
            # 绘制文本框边界
            cv2.polylines(img_vis, [box], isClosed=True,
                          color=self.cfg['color_boundary'], thickness=2)
            
            # 绘制编号标签
            # 选择左上角点作为标签位置
            pt_idx = np.argmin(np.sum(box, axis=1))
            t_x, t_y = box[pt_idx]
            label = str(idx)
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 绘制标签背景
            cv2.rectangle(img_vis,
                          (int(t_x), int(t_y - text_height - 4)),
                          (int(t_x + text_width), int(t_y)),
                          self.cfg['color_boundary'], -1)
            
            # 绘制标签文字
            cv2.putText(img_vis, label, (int(t_x), int(t_y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存结果图像
        save_path = self.img_path.replace('.png', '_ocr_result.png').replace('.jpg', '_ocr_result.jpg')
        cv2.imwrite(save_path, img_vis)
        print(f"💾 [保存] 结果图像已保存至: {save_path}")

        # 显示图像
        cv2.imshow("OCR Extractor - Detection Results", img_vis)
        print("🏁 [等待] 请按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图表文本OCR提取工具 (Chart Text OCR Extractor)")

    # 基础输入
    parser.add_argument("--img_path", type=str, required=True,
                        help="输入图像的路径 (支持 .jpg, .png)。")

    # OCR核心参数
    parser.add_argument("--db_thresh", type=float, default=0.3,
                        help="文本检测阈值，越低检测越敏感。")
    parser.add_argument("--box_thresh", type=float, default=0.5,
                        help="文本框阈值，控制文本框的生成。")
    parser.add_argument("--unclip_ratio", type=float, default=1.2,
                        help="文本框扩展比例，用于处理紧凑文本。")
    parser.add_argument("--rec_thresh", type=float, default=0.6,
                        help="文本识别置信度阈值，低于此值的识别结果将被过滤。")

    # 后处理开关
    parser.add_argument("--enable_split_numbers", action="store_true", default=True,
                        help="启用合并数字拆分，针对粘连的数字序列。")
    parser.add_argument("--filter_geometric_symbols", action="store_true", default=True,
                        help="过滤几何符号（如短横线、图例标记）。")
    parser.add_argument("--remove_vertical_text", action="store_true", default=True,
                        help="过滤垂直文本（通常不是坐标轴标签）。")
    parser.add_argument("--max_text_angle", type=float, default=30.0,
                        help="最大文本角度，用于过滤非水平文本。")

    # 系统参数
    parser.add_argument("--use_cuda", action="store_true",
                        help="启用CUDA加速。")
    parser.add_argument("--no_plot", action="store_true",
                        help="静默模式：不弹出显示窗口，适合命令行批处理或服务器运行。")
    parser.add_argument("--silent", action="store_true",
                        help="静默模式：不输出控制台日志。")

    args = parser.parse_args()

    # 执行主程序
    try:
        extractor = OCRExtractor(args.img_path)

        results = extractor.process(
            db_thresh=args.db_thresh,
            box_thresh=args.box_thresh,
            unclip_ratio=args.unclip_ratio,
            rec_thresh=args.rec_thresh,
            enable_split_numbers=args.enable_split_numbers,
            filter_geometric_symbols=args.filter_geometric_symbols,
            remove_vertical_text=args.remove_vertical_text,
            max_text_angle=args.max_text_angle,
            use_cuda=args.use_cuda,
            no_plot=args.no_plot,
            verbose=not args.silent
        )

        if not args.silent:
            print(f"\n✅ OCR提取完成，共提取到 {len(results)} 个文本区域")

    except Exception as e:
        print(f"❌ 程序执行出错: {str(e)}")
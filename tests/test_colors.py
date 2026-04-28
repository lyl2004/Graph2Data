import cv2
import numpy as np
import argparse
import os
import math
from typing import List, Dict, Tuple, Optional

class ColorExtractor:
    """
    科学图表颜色提取器 (Scientific Chart Color Extractor)

    该类专用于从科学论文图表中提取数据曲线的颜色。
    针对性解决了以下技术难点：
    1. 细线条识别：防止 1px 宽度的线条在降采样中丢失。
    2. 虚线/点线提取：通过极小面积阈值捕获断续的线条片段。
    3. 抗锯齿干扰：通过分离亮度(L)与色度(ab)权重的聚类算法，合并边缘杂色。
    """

    def __init__(self, img_path: str = None, img_array: np.ndarray = None):
        """
        初始化提取器实例。
        Args:
            img_path (str): 图像文件的系统路径。如果提供img_array，则忽略此参数。
            img_array (np.ndarray): 直接传入的图像数组（BGR格式）。
        """
        if img_array is not None:
            self.img = img_array.copy()
            self.img_path = "memory_image"
        elif img_path is not None:
            self.img_path = img_path
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"错误: 找不到文件路径: {img_path}")
            self.img = cv2.imread(img_path)
        else:
            raise ValueError("必须提供img_path或img_array")
        if self.img is None:
            raise ValueError("错误: 图像读取失败")
            
        self.h, self.w = self.img.shape[:2]
        self.total_pixels = self.h * self.w
        
        # 色彩空间转换: BGR -> Lab
        # 选择 Lab 空间是因为其具有感知均匀性 (Perceptual Uniformity)，
        # 相比 RGB 空间，Lab 空间计算的欧氏距离更符合人眼对色差的感知。
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)
        
        # 转换为 float32 以避免在后续计算距离时发生溢出或精度丢失
        self.img_lab_float = self.img_lab.astype(np.float32)

    def process(self, 
                step: int = 1,                 
                diff: int = 15,                
                min_area: int = 2,             
                min_l: int = 10, max_l: int = 240,    
                min_chroma: int = 10,          
                merge_diff_l: int = 60,        
                merge_diff_ab: int = 60,       
                min_ratio: float = 0.0003,      
                max_ratio: float = 0.90,         
                range_l: int = 60, 
                range_ab: int = 15, 
                verbose: bool = True, 
                visualize: bool = True) -> List[Dict]:
        """
        执行核心颜色提取流程。

        流程包括：预过滤 -> 漫水填充连通域提取 -> 颜色聚类合并 -> 结果后处理。

        Args:
            step (int): 图像扫描步长。建议设为 1 以确保捕获最细微的线条。
            diff (int): 漫水填充 (FloodFill) 算法的生长容差。
            min_area (int): 连通域最小像素面积。针对点状虚线需设为极低值 (如 2)。
            
            min_l (int): 预过滤最小亮度 (0-255)。用于剔除黑色文字/轴线。
            max_l (int): 预过滤最大亮度 (0-255)。用于剔除白色背景。
            min_chroma (int): 预过滤最小色度。用于剔除灰色网格线。
            
            merge_diff_l (int): 颜色合并阶段的亮度容差。允许同色系深浅合并。
            merge_diff_ab (int): 颜色合并阶段的色度容差。用于严格区分不同色相。
            
            min_ratio (float): 最终保留颜色的最小面积占比。用于过滤抗锯齿噪点。
            max_ratio (float): 最终保留颜色的最大面积占比。用于防止背景误检。
            
            range_l (int): 可视化提取图层时的亮度宽容度。
            range_ab (int): 可视化提取图层时的色度宽容度。
            verbose (bool): 是否打印控制台日志。
            visualize (bool): 是否显示结果窗口。

        Returns:
            List[Dict]: 提取到的颜色列表，包含 RGB、Lab、面积占比等信息。
        """
        
        if verbose: print(f"🚀 [启动] 正在分析图像: {self.img_path}")

        # ==========================================
        # 阶段 1: 预过滤 (Pre-filtering)
        # 目的: 生成“忽略掩膜”，快速剔除背景、文字和无关网格。
        # ==========================================
        L_c = self.img_lab[:, :, 0]
        # OpenCV Lab 中 a, b 均值偏移量为 128
        a_c = self.img_lab_float[:, :, 1] - 128
        b_c = self.img_lab_float[:, :, 2] - 128
        # 计算色度 (Chroma) = sqrt(a^2 + b^2)
        chroma = np.sqrt(a_c**2 + b_c**2)
        
        # 构建布尔掩膜: 亮度过低(黑) OR 亮度过高(白) OR 色度过低(灰)
        mask_ignore = (L_c < min_l) | (L_c > max_l) | (chroma < min_chroma)

        # ==========================================
        # 阶段 2: 漫水填充 (FloodFill Segmentation)
        # 目的: 从未被忽略的区域提取连通分量。
        # ==========================================
        if verbose: print(f"🌊 [步骤 1] 执行漫水填充提取 (步长: {step})...")
        
        visited = np.zeros((self.h, self.w), dtype=bool)
        visited[mask_ignore] = True
        
        # FloodFill 掩膜必须比原图长宽各多 2 个像素 (OpenCV 规范)
        mask_template = np.zeros((self.h + 2, self.w + 2), np.uint8)
        raw_regions = []
        
        # 初始化可视化底图 (全黑背景，用于高对比度显示边缘)
        img_regions_vis = np.zeros_like(self.img) if visualize else None
            
        for y in range(0, self.h, step):
            for x in range(0, self.w, step):
                if visited[y, x]: continue

                mask = mask_template.copy()
                # flags高8位: 填充值(255) | 低位: 4连通 + 固定范围模式 + 仅填充掩膜
                flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
                tolerance = (diff, diff, diff)

                area, _, _, _ = cv2.floodFill(self.img_lab, mask, (x, y), (0,0,0), tolerance, tolerance, flags)
                
                # 提取 ROI 掩膜
                region_mask = mask[1:self.h+1, 1:self.w+1].astype(bool)
                visited[region_mask] = True

                # 面积筛选
                if area >= min_area:
                    # 计算该连通域的平均 Lab 颜色
                    mean_lab = cv2.mean(self.img_lab, mask=region_mask.astype(np.uint8))[:3]
                    
                    # 二次校验: 确保区域平均值不在被忽略的范围内 (防止边缘溢出)
                    l_val = mean_lab[0]
                    c_val = math.sqrt((mean_lab[1]-128)**2 + (mean_lab[2]-128)**2)
                    
                    if l_val >= min_l and l_val <= max_l and c_val >= min_chroma:
                        raw_regions.append({'lab': np.array(mean_lab), 'area': area})
                        
                        # 绘制轮廓用于可视化
                        if visualize and img_regions_vis is not None:
                            lab_pix = np.uint8([[mean_lab]])
                            bgr_pix = cv2.cvtColor(lab_pix, cv2.COLOR_Lab2BGR)[0][0]
                            real_color = (int(bgr_pix[0]), int(bgr_pix[1]), int(bgr_pix[2]))
                            
                            contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(img_regions_vis, contours, -1, real_color, 2)

        if verbose: print(f"✅ [步骤 1 完成] 初步提取到 {len(raw_regions)} 个连通区域")

        # ==========================================
        # 阶段 3: 颜色聚类合并 (Color Merging)
        # 目的: 合并属于同一物体但因抗锯齿或断裂分离的区域。
        # ==========================================
        if verbose: print(f"🎨 [步骤 2] 执行加权颜色合并...")
        
        # 按面积降序排序，确保大面积主色作为聚类基准
        raw_regions.sort(key=lambda x: x['area'], reverse=True)
        merged_palette = []

        for region in raw_regions:
            curr_lab = region['lab']
            curr_area = region['area']
            
            matched = False
            for p in merged_palette:
                target_lab = p['lab']
                
                # 分离计算 L(亮度) 和 ab(色度) 的差异
                delta_l = abs(curr_lab[0] - target_lab[0])
                delta_a = curr_lab[1] - target_lab[1]
                delta_b = curr_lab[2] - target_lab[2]
                delta_ab = math.sqrt(delta_a**2 + delta_b**2)
                
                # 合并判据:
                # 必须同时满足亮度阈值和色度阈值。
                # 通常设置宽松的 L 阈值 (允许深浅变化) 和严格的 ab 阈值 (区分不同颜色)
                if delta_l < merge_diff_l and delta_ab < merge_diff_ab:
                    # 使用面积加权平均法更新颜色中心，保证大面积区域主导颜色值
                    new_area = p['area'] + curr_area
                    p['lab'] = (p['lab'] * p['area'] + curr_lab * curr_area) / new_area
                    p['area'] = new_area
                    p['count'] += 1
                    
                    # 更新显示的 RGB 值
                    lab_pix = np.uint8([[p['lab']]])
                    bgr_pix = cv2.cvtColor(lab_pix, cv2.COLOR_Lab2BGR)[0][0]
                    p['rgb'] = (int(bgr_pix[2]), int(bgr_pix[1]), int(bgr_pix[0]))
                    matched = True
                    break
            
            if not matched:
                lab_pix = np.uint8([[curr_lab]])
                bgr_pix = cv2.cvtColor(lab_pix, cv2.COLOR_Lab2BGR)[0][0]
                rgb_val = (int(bgr_pix[2]), int(bgr_pix[1]), int(bgr_pix[0]))
                
                merged_palette.append({
                    'lab': curr_lab,
                    'rgb': rgb_val,
                    'area': curr_area,
                    'count': 1,
                    'ratio': curr_area / self.total_pixels
                })

        # ==========================================
        # 阶段 4: 结果后处理过滤 (Post-Filtering)
        # 目的: 根据面积占比剔除噪点。
        # ==========================================
        final_palette = []
        if verbose: print(f"🧹 [步骤 3] 过滤杂色与背景 (阈值: {min_ratio:.4%})...")
        
        for p in merged_palette:
            ratio = p['area'] / self.total_pixels
            p['ratio'] = ratio 
            
            if ratio >= min_ratio and ratio <= max_ratio:
                final_palette.append(p)
            else:
                if verbose: print(f"   [剔除] RGB:{p['rgb']} 占比:{ratio:.4%} (不满足阈值条件)")

        final_palette.sort(key=lambda x: x['area'], reverse=True)
        
        if verbose: print(f"✅ [全部完成] 最终锁定 {len(final_palette)} 种有效颜色")

        # ==========================================
        # 阶段 5: 可视化输出 (Visualization)
        # ==========================================
        if visualize:
            self._visualize_results(final_palette, img_regions_vis, mask_ignore, range_l, range_ab)
        
        return final_palette

    def _visualize_results(self, palette: List[Dict], img_regions: np.ndarray, mask_ignore: np.ndarray, range_l: int, range_ab: int):
        """
        生成可视化分析窗口。
        包含: 原始图、预过滤掩膜、边缘连通域、调色板列表、图层分层提取结果。
        """
        print(f"🖼️ [UI] 正在生成可视化界面...")
        
        card_h, card_w = 60, 500
        
        # 1. 绘制调色板 (左侧列表)
        # 关键修正: 使用 np.full 填充 255 并指定 dtype=np.uint8，防止 OpenCV 显示为全黑
        if not palette:
            palette_img = np.full((100, card_w, 3), 255, dtype=np.uint8)
            cv2.putText(palette_img, "未检测到有效颜色", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            palette_img = np.full((len(palette) * card_h, card_w, 3), 255, dtype=np.uint8)
            for idx, p in enumerate(palette):
                y0, y1 = idx * card_h, (idx + 1) * card_h
                color_bgr = (p['rgb'][2], p['rgb'][1], p['rgb'][0])
                
                # 颜色块
                cv2.rectangle(palette_img, (0, y0), (100, y1), color_bgr, -1)
                
                # 文字详情
                text_info = f"ID:{idx} Area:{p['area']} ({p['ratio']:.2%})"
                text_lab = f"L:{int(p['lab'][0])} a:{int(p['lab'][1])} b:{int(p['lab'][2])}"
                cv2.putText(palette_img, text_info, (120, y0+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.putText(palette_img, text_lab, (120, y0+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1)

        # 2. 绘制分层提取网格 (右侧概览)
        num_colors = len(palette)
        if num_colors == 0:
             layers_grid = np.full((300, 300, 3), 255, dtype=np.uint8)
        else:
            cols = 3
            rows = math.ceil(num_colors / cols)
            th_h, th_w = self.h // 2, self.w // 2 
            grid_h = max(rows * th_h, 100)
            grid_w = max(cols * th_w, 100)
            
            # 初始化白色画布
            layers_grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
            
            # 使用 NumPy 向量化操作加速全图距离计算
            L_data = self.img_lab_float[:,:,0]
            a_data = self.img_lab_float[:,:,1]
            b_data = self.img_lab_float[:,:,2]

            for idx, p in enumerate(palette):
                r, c = idx // cols, idx % cols
                y_off, x_off = r * th_h, c * th_w
                
                target_lab = p['lab']
                diff_L = np.abs(L_data - target_lab[0])
                diff_ab = np.sqrt((a_data - target_lab[1])**2 + (b_data - target_lab[2])**2)
                
                # 生成提取掩膜
                mask = (diff_L < range_l) & (diff_ab < range_ab)
                
                # 背景置白，前景填色
                layer_view = np.full_like(self.img, 255)
                layer_view[mask] = self.img[mask]
                
                thumb = cv2.resize(layer_view, (th_w, th_h))
                cv2.rectangle(thumb, (0,0), (th_w-1, th_h-1), (200,200,200), 1)
                cv2.putText(thumb, f"ID {idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                
                end_y = min(y_off+th_h, grid_h)
                end_x = min(x_off+th_w, grid_w)
                layers_grid[y_off:end_y, x_off:end_x] = thumb[:end_y-y_off, :end_x-x_off]

        # 3. 显示窗口
        img_preview = self.img.copy()
        img_preview[mask_ignore] = (255, 255, 255)
        
        cv2.imshow("1. Original Image", self.img)
        cv2.imshow("2. Pre-filter Mask", img_preview)
        cv2.imshow("3. Detected Regions (Edges)", img_regions) 
        cv2.imshow("4. Color Palette", palette_img)
        cv2.imshow("5. Layer Extraction", layers_grid)
        
        print("🏁 [等待] 请按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="科学图表颜色提取工具 (Scientific Chart Color Extractor)")
    
    # 基础输入
    parser.add_argument("--img_path", type=str, default="tests/test2", 
                        help="输入图像的路径 (支持 .jpg, .png)。")
    
    # 核心算法参数
    parser.add_argument("--step", type=int, default=1, 
                        help="图像扫描步长。设为 1 以确保不漏掉 1px 宽度的细线。")
    parser.add_argument("--diff", type=int, default=15, 
                        help="漫水填充 (FloodFill) 的单次生长颜色容差。")
    parser.add_argument("--min_area", type=int, default=2, 
                        help="连通域最小面积。针对虚线或点状线，必须设为极小值 (如 2)。")

    # 预过滤 (背景剔除)
    parser.add_argument("--min_l", type=int, default=10, 
                        help="预过滤最小亮度。用于剔除黑色文字、轴线等。")
    parser.add_argument("--max_l", type=int, default=240, 
                        help="预过滤最大亮度。用于剔除白色背景。")
    parser.add_argument("--min_chroma", type=int, default=20, 
                        help="预过滤最小色度。用于剔除灰色网格线。")

    # 颜色合并 (聚类)
    parser.add_argument("--merge_diff_l", type=int, default=80, 
                        help="颜色合并时的亮度容差 (L通道)。允许较大的亮度差异以合并同一颜色的深浅变化（如抗锯齿边缘）。")
    parser.add_argument("--merge_diff_ab", type=int, default=20, 
                        help="颜色合并时的色度容差 (ab通道)。用于严格区分不同色相 (如红色与紫色)。")

    # 结果过滤
    parser.add_argument("--min_ratio", type=float, default=0.001, 
                        help="最小面积占比阈值 (例如 0.0003 代表 0.03%)。用于保留点状虚线并过滤抗锯齿边缘的杂色。")
    parser.add_argument("--max_ratio", type=float, default=0.90, 
                        help="最大面积占比阈值。用于剔除可能残留的大面积背景区域。")

    # 可视化设置
    parser.add_argument("--range_l", type=int, default=60, 
                        help="可视化最终图层时的亮度宽容度。")
    parser.add_argument("--range_ab", type=int, default=15, 
                        help="可视化最终图层时的色度宽容度。")
    
    # 运行模式
    parser.add_argument("--silent", action="store_true", help="静默模式：不输出控制台日志。")
    parser.add_argument("--no_gui", action="store_true", help="无界面模式：不显示弹窗图片。")

    args = parser.parse_args()
    
    # 路径自动补全
    if not os.path.exists(args.img_path):
        if os.path.exists(args.img_path + ".png"): args.img_path += ".png"
        elif os.path.exists(args.img_path + ".jpg"): args.img_path += ".jpg"

    # 执行主程序
    extractor = ColorExtractor(args.img_path)
    
    colors = extractor.process(
        step=args.step,
        diff=args.diff,
        min_area=args.min_area,
        min_l=args.min_l,
        max_l=args.max_l,
        min_chroma=args.min_chroma,
        merge_diff_l=args.merge_diff_l,
        merge_diff_ab=args.merge_diff_ab,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        range_l=args.range_l,
        range_ab=args.range_ab,
        verbose=not args.silent,
        visualize=not args.no_gui
    )

    if not args.silent:
        print("\n=== 最终提取结果列表 ===")
        for idx, c in enumerate(colors):
            print(f"ID: {idx} | RGB: {c['rgb']} | 占比: {c['ratio']:.4%}")
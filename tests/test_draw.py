#!/usr/bin/env python3
"""
tests/test_draw.py

为重叠线条补全项目生成测试用图：
  - img_curve.png      : 平滑曲线图，16条曲线，8色4线型（含灰色/绿色细虚线）
  - img_broken.png     : 折线图，16条折线，8色4种节点标记，全部实线（无虚线）
  - cor_curve.png      : 曲线图色卡，展示所有颜色+线型组合
  - cor_broken.png     : 折线图色卡，展示所有颜色+标记组合

图像不包含坐标轴、图例、图表名，仅含浅灰色虚线网格。
三阶段划分严格，线条密度、重叠程度、走势变化均按需求设计。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 固定随机种子以确保可重现性
np.random.seed(42)

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
# 8种颜色：黑色、灰色、红色、蓝色、绿色、橙色、紫色、棕色
COLORS = ['black', 'gray', 'red', 'blue', 'green', 'orange', 'purple', 'brown']

# 曲线图线型分配（16条曲线：每种颜色2种线型，必须包含灰色细虚线、绿色细虚线）
CURVE_STYLES = [
    ('black', '-'), ('black', '--'),
    ('gray', '-'),  ('gray', '--'),       # 灰色细虚线
    ('red', '-'),   ('red', '-.'),
    ('blue', '-'),  ('blue', ':'),
    ('green', '-'), ('green', '--'),      # 绿色细虚线
    ('orange', '-'),('orange', '-.'),
    ('purple', '-'),('purple', ':'),
    ('brown', '-'), ('brown', '--'),
]

# 折线图样式分配（16条折线：每种颜色2种节点标记，线型全部实线）
BROKEN_MARKERS = [
    ('black', 'o'), ('black', 's'),
    ('gray', '^'),  ('gray', 'D'),
    ('red', 'v'),   ('red', '<'),
    ('blue', '>'),  ('blue', 'p'),
    ('green', '*'), ('green', 'h'),
    ('orange', 'H'),('orange', '+'),
    ('purple', 'x'),('purple', 'd'),
    ('brown', '|'), ('brown', '_'),
]

# ----------------------------------------------------------------------
# 数据生成：折线图（离散节点，直线连接，带标记）
# ----------------------------------------------------------------------
def generate_broken_data():
    """
    生成折线图的x坐标和16条折线的y值。
    节点分布：区域1: 8个点 (0~10) → 7次偏折，前2次完全重叠
            区域2: 11个点 (10~20) → 10次走势变化
            区域3: 11个点 (20~30) → 10次走势变化
    返回:
        x: (28,) 节点横坐标
        y_list: list of (28,) 每条折线的纵坐标
        group_assign: 每条线所属股(0~3)，用于区域2
    """
    # ---------- 节点定义 ----------
    x1 = np.linspace(0, 10, 8)      # 8点, 7段
    x2 = np.linspace(10, 20, 11)    # 11点, 10段
    x3 = np.linspace(20, 30, 11)    # 11点, 10段
    x = np.concatenate([x1, x2[1:], x3[1:]])  # 去重端点，总28点

    n_curves = 16
    # ---------- 区域2：4个股中心趋势（硬编码，确保10次以上走势变化）--------
    trends = {
        0: np.array([1.5, 1.8, 1.4, 2.0, 1.6, 2.2, 1.9, 2.5, 2.1, 2.7, 2.4]),
        1: np.array([1.0, 0.7, 1.2, 0.8, 1.3, 0.9, 1.4, 1.0, 1.5, 1.1, 1.6]),
        2: np.array([0.8, 1.1, 0.6, 1.3, 0.9, 1.5, 1.2, 1.8, 1.4, 2.0, 1.7]),
        3: np.array([0.5, 0.9, 0.4, 1.0, 0.6, 1.2, 0.8, 1.4, 1.0, 1.6, 1.3]),
    }
    # 分配16条线到4个股：股0(5条), 股1(4条), 股2(4条), 股3(3条)
    group_sizes = [5, 4, 4, 3]
    group_assign = []
    for g, sz in enumerate(group_sizes):
        group_assign.extend([g] * sz)

    # ---------- 每条线的个性化参数 ----------
    # 区域1终点偏移 delta_i1 (x=10处)，范围0~0.5，有密有疏
    delta1 = np.random.uniform(0, 0.5, n_curves)
    # 区域2终点偏移 delta_i2 (x=20处)，范围-0.3~0.3
    delta2 = np.random.uniform(-0.3, 0.3, n_curves)
    # 区域2内部波动幅度
    amp2 = np.random.uniform(0.05, 0.15, n_curves)
    # 区域2波动相位
    phi2 = np.random.uniform(0, 2*np.pi, n_curves)
    # 区域3波动相位
    phi3 = np.random.uniform(0, 2*np.pi, n_curves)

    # ---------- 存储所有线的y值 ----------
    y_list = []
    y2_last_all = []

    # 第一遍：生成区域1和区域2，记录区域2终点值
    for i in range(n_curves):
        g = group_assign[i]

        # ----- 区域2 y2 (11个点) -----
        base_y2 = trends[g].copy()
        # 偏移函数: 在x=10处为delta1[i], x=20处为delta2[i], 内部正弦波动
        t = (x2 - 10) / 10  # 0~1
        offset2 = delta1[i] + (delta2[i] - delta1[i]) * t \
                  + amp2[i] * np.sin(np.pi * t) * np.sin(2 * np.pi * 0.5 * (x2 - 10) + phi2[i])
        y2 = base_y2 + offset2
        y2[0] = base_y2[0] + delta1[i]   # 确保端点精确
        y2[-1] = base_y2[-1] + delta2[i]

        # ----- 区域1 y1 (8个点) -----
        # 前7个点固定走势，第8个点(x=10)调整为与y2[0]连续
        base_y1_part = np.array([0.0, 0.6, 0.2, 0.9, 0.4, 1.1, 0.7])
        target_last = y2[0] - delta1[i]   # 使y1[-1] + delta1[i] = y2[0]
        base_y1 = np.concatenate([base_y1_part, [target_last]])

        # 区域1偏移: 0-4为0, 4-10线性增加至delta1[i]
        offset1 = np.zeros_like(x1)
        mask = x1 >= 4
        offset1[mask] = delta1[i] * (x1[mask] - 4) / 6.0
        y1 = base_y1 + offset1

        # ----- 暂存区域2终点值，用于区域3 -----
        y2_last_all.append(y2[-1])
        y_list.append((y1, y2, None))   # 第三项占位

    # ---------- 区域3：确保互不遮挡 ----------
    # 根据y2[-1]排序，分配不同斜率（小→小，大→大），并加微小波动
    y2_last_sorted_idx = np.argsort(y2_last_all)
    k_values = np.linspace(0.02, 0.18, n_curves)  # 斜率范围
    k_assigned = np.zeros(n_curves)
    for rank, idx in enumerate(y2_last_sorted_idx):
        k_assigned[idx] = k_values[rank]

    # 第二遍：生成区域3并拼接完整y
    for i in range(n_curves):
        y1, y2, _ = y_list[i]
        # 区域3线性趋势 + 微小正弦波动
        y3 = y2[-1] + k_assigned[i] * (x3 - 20) \
              + 0.04 * np.sin(2 * np.pi * 0.6 * (x3 - 20) + phi3[i])
        # 拼接: y1(8) + y2[1:] (去掉x=10重复点) + y3[1:] (去掉x=20重复点)
        y_full = np.concatenate([y1, y2[1:], y3[1:]])
        y_list[i] = y_full

    return x, y_list, group_assign


# ----------------------------------------------------------------------
# 数据生成：平滑曲线图（密集点，连续曲线）
# ----------------------------------------------------------------------
def generate_curve_data():
    """
    生成平滑曲线图的x坐标和16条曲线的y值。
    使用密集采样，三阶段特征与折线图类似，但通过连续函数实现平滑。
    区域1前40%几乎完全重叠，后60%逐渐散开；区域2分4股；区域3互不遮挡且保持曲线波动。
    """
    x = np.linspace(0, 30, 800)  # 密集点
    n_curves = 16

    # 分组与折线图保持一致
    group_sizes = [5, 4, 4, 3]
    group_assign = []
    for g, sz in enumerate(group_sizes):
        group_assign.extend([g] * sz)

    # ---------- 区域2：4个股中心趋势（连续函数）---------
    def trend0(x2):
        return 1.5 + 0.2*(x2-10) + 0.3*np.sin(0.6*(x2-10)) + 0.1*np.cos(0.9*(x2-10))
    def trend1(x2):
        return 1.0 - 0.1*(x2-10) + 0.25*np.cos(0.7*(x2-10)) + 0.1*np.sin(1.1*(x2-10))
    def trend2(x2):
        return 0.8 + 0.05*(x2-10) + 0.2*np.sin(0.5*(x2-10)+1) + 0.15*np.cos(0.8*(x2-10))
    def trend3(x2):
        return 0.5 + 0.03*(x2-10) + 0.3*np.sin(0.4*(x2-10)+2) - 0.1*np.cos(0.7*(x2-10))
    trends = [trend0, trend1, trend2, trend3]

    # ---------- 每条线的个性化参数 ----------
    delta1 = np.random.uniform(0, 0.6, n_curves)   # x=10处偏移
    delta2 = np.random.uniform(-0.4, 0.4, n_curves) # x=20处偏移
    amp2 = np.random.uniform(0.1, 0.25, n_curves)  # 区域2波动幅度
    phi2 = np.random.uniform(0, 2*np.pi, n_curves)
    phi3 = np.random.uniform(0, 2*np.pi, n_curves)

    # 存储每条线的y值及区域2终点
    y_list = []
    y2_last_all = []

    # 第一遍：生成整体趋势，记录x=20处的y值
    for i in range(n_curves):
        g = group_assign[i]
        # 区域2 (10~20)
        mask2 = (x >= 10) & (x < 20)
        x2 = x[mask2]
        base_y2 = trends[g](x2)
        # 偏移：端点连续，内部正弦波动
        t2 = (x2 - 10) / 10
        offset2 = delta1[i] + (delta2[i] - delta1[i]) * t2 \
                  + amp2[i] * np.sin(np.pi * t2) * np.sin(2 * np.pi * 0.5 * (x2 - 10) + phi2[i])
        y2 = base_y2 + offset2
        # 确保端点精确
        y2[0] = trends[g](10) + delta1[i]
        y2[-1] = trends[g](20) + delta2[i]

        # 区域1 (0~10)
        mask1 = x < 10
        x1 = x[mask1]
        # 基线：正弦+线性，确保有足够走势变化
        base_y1 = 0.2 * np.sin(1.8 * x1) + 0.05 * x1 + 0.5
        # 偏移：0-4几乎为0，4-10线性增加至delta1[i]，且x=10处连续
        offset1 = np.zeros_like(x1)
        mask1_4 = x1 >= 4
        offset1[mask1_4] = delta1[i] * (x1[mask1_4] - 4) / 6.0
        # 调整基线使x=10处与y2[0]连续
        y1_target = y2[0] - delta1[i]
        base_y1[-1] = y1_target  # 强制连续
        y1 = base_y1 + offset1

        # 暂存区域2终点
        y2_last_all.append(y2[-1])
        y_list.append((y1, y2, None, mask1, mask2))  # 暂缺区域3

    # 分配区域3斜率，保证不交叉
    y2_last_sorted_idx = np.argsort(y2_last_all)
    k_values = np.linspace(0.01, 0.15, n_curves)
    k_assigned = np.zeros(n_curves)
    for rank, idx in enumerate(y2_last_sorted_idx):
        k_assigned[idx] = k_values[rank]

    # 第二遍：生成区域3并拼接
    final_y_list = []
    for i in range(n_curves):
        y1, y2, _, mask1, mask2 = y_list[i]
        mask3 = x >= 20
        x3 = x[mask3]
        # 线性趋势 + 小幅波动，保证曲线特性且不交叉
        y3 = y2[-1] + k_assigned[i] * (x3 - 20) \
              + 0.03 * np.sin(2 * np.pi * 0.7 * (x3 - 20) + phi3[i]) \
              + 0.02 * np.cos(2 * np.pi * 0.4 * (x3 - 20))
        # 整体拼接
        y_full = np.concatenate([y1, y2, y3])
        final_y_list.append(y_full)

    return x, final_y_list, group_assign


# ----------------------------------------------------------------------
# 绘图函数
# ----------------------------------------------------------------------
def setup_axes(ax):
    """隐藏坐标轴、刻度、脊柱，保留浅灰色虚线网格"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
    ax.set_facecolor('white')

def draw_curve():
    """绘制平滑曲线图：tests/img_curve.png"""
    x, y_list, _ = generate_curve_data()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    setup_axes(ax)

    for i, y in enumerate(y_list):
        color, linestyle = CURVE_STYLES[i]
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)

    # 保存
    save_path = os.path.join(os.path.dirname(__file__), 'img_curve.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"已保存: {save_path}")

def draw_broken():
    """绘制折线图：tests/img_broken.png (无虚线，带节点标记)"""
    x, y_list, _ = generate_broken_data()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    setup_axes(ax)

    for i, y in enumerate(y_list):
        color, marker = BROKEN_MARKERS[i]
        # 折线图全部实线，节点标记区分，标记大小适中，每点都标
        ax.plot(x, y, color=color, linestyle='-', marker=marker,
                markersize=2.5, markevery=1, linewidth=0.8, alpha=0.9)

    save_path = os.path.join(os.path.dirname(__file__), 'img_broken.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"已保存: {save_path}")

def draw_cor_curve():
    """绘制曲线图色卡：tests/cor_curve.png，展示所有颜色+线型"""
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 17)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    y_pos = 16
    for i, (color, linestyle) in enumerate(CURVE_STYLES):
        # 绘制示例线段
        ax.plot([0.1, 0.5], [y_pos-0.2, y_pos-0.2],
                color=color, linestyle=linestyle, linewidth=2)
        # 标注颜色+线型
        ax.text(0.55, y_pos-0.25, f'{color}, {linestyle}',
                fontsize=8, verticalalignment='center')
        y_pos -= 1

    ax.set_title('Curve Color & Linestyle Reference', fontsize=12, pad=10)
    save_path = os.path.join(os.path.dirname(__file__), 'cor_curve.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已保存: {save_path}")

def draw_cor_broken():
    """绘制折线图色卡：tests/cor_broken.png，展示所有颜色+标记"""
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 17)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    y_pos = 16
    for i, (color, marker) in enumerate(BROKEN_MARKERS):
        # 绘制带标记的线段
        ax.plot([0.1, 0.5], [y_pos-0.2, y_pos-0.2],
                color=color, linestyle='-', marker=marker,
                markersize=6, markevery=1, linewidth=1.5)
        ax.text(0.55, y_pos-0.25, f'{color}, marker={marker}',
                fontsize=8, verticalalignment='center')
        y_pos -= 1

    ax.set_title('Broken Line Color & Marker Reference', fontsize=12, pad=10)
    save_path = os.path.join(os.path.dirname(__file__), 'cor_broken.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已保存: {save_path}")

# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    # 确保输出目录存在（即tests目录）
    os.makedirs(os.path.dirname(__file__), exist_ok=True)

    # 设置Seaborn风格（白色背景，网格自定义）
    sns.set_theme(style='whitegrid')
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = 'lightgray'
    plt.rcParams['grid.linewidth'] = 0.5

    print("开始生成测试图像...")
    draw_curve()
    draw_broken()
    draw_cor_curve()
    draw_cor_broken()
    print("所有图像生成完毕。")

if __name__ == '__main__':
    main()
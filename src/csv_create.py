import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import argparse
import re
from glob import glob

# -------------------- 配置 --------------------
SAVE_DIR = './temp'
os.makedirs(SAVE_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  # 仅用于保证可复现性，不影响曲线形状

COLORS_TAB10 = plt.cm.tab10.colors
COLORS_SET1 = plt.cm.Set1.colors

# 存储元数据
metadata_list = []

def save_figure(fig, filename, dpi=150):
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

# -------------------- 辅助函数（无噪声版本）--------------------
def generate_line_data(x, base_func, n_curves=5):
    """生成多条确定性折线（无噪声）"""
    y_list = []
    params_list = []
    for i in range(n_curves):
        phase_shift = i * 0.8
        amplitude = 0.8 + 0.4 * np.sin(i)
        y = amplitude * base_func(x + phase_shift)
        y_list.append(y)
        params_list.append(f'phase={phase_shift:.2f}, amp={amplitude:.3f}')
    return y_list, params_list

# -------------------- 曲线生成函数（参数化，便于校验重建）--------------------
def gen_curve_1(x):
    """单条黑色曲线，三段频率渐增"""
    y = np.zeros_like(x)
    mask1 = x <= 60
    mask2 = (x > 60) & (x <= 120)
    mask3 = x > 120
    y[mask1] = 50 + 30 * np.sin(2 * np.pi * x[mask1] / 60)
    x_mid = x[mask2] - 60
    y[mask2] = 50 + 30 * np.sin(2 * np.pi * 1 + 2 * np.pi * 2 * x_mid / 60)
    x_high = x[mask3] - 120
    y[mask3] = 50 + 30 * np.sin(2 * np.pi * 5 + 2 * np.pi * 4 * x_high / 60)
    return y, ['black']

def gen_curve_2(x):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    y_list = []
    params = []
    for i in range(4):
        phase = i * np.pi / 2
        amp = 20 + 5 * i
        freq = 0.02 + 0.005 * i
        y = amp * np.sin(freq * x + phase) + 20 * np.cos(0.01 * x)
        y_list.append(y)
        params.append(f'amp={amp:.2f}, freq={freq:.4f}, phase={phase:.2f}')
    return y_list, colors, params

def gen_curve_3(x):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    y_list = []
    params = []
    for i in range(4):
        phase = i * np.pi / 2
        amp = 30 + 10 * i
        freq = 0.1 + 0.03 * i
        y = amp * np.sin(freq * x + phase) * np.cos(0.015 * x) + 20 * np.sin(0.2 * x)
        y_list.append(y)
        params.append(f'amp={amp:.2f}, freq={freq:.4f}, phase={phase:.2f}')
    return y_list, colors, params

def gen_curve_4(x):
    colors = COLORS_SET1
    y_list = []
    params = []
    for i in range(8):
        phase = i * 0.8
        amp = 2000 + 500 * i
        freq = 0.002 + 0.0005 * i
        y = amp * np.sin(freq * x + phase) + 1000 * np.cos(0.001 * x)
        y_list.append(y)
        params.append(f'amp={amp:.2f}, freq={freq:.6f}, phase={phase:.2f}')
    return y_list, colors, params

# -------------------- create 模式：生成图像 --------------------
def create_figures():
    print("Generating line charts...")
    x_line = np.linspace(0, 10, 20)

    line_functions = [
        ('sin(x)', lambda x: np.sin(x), 'Line 1: Sine Family'),
        ('cos(x)*exp(-x/10)', lambda x: np.cos(x) * np.exp(-x/10), 'Line 2: Damped Cosine'),
        ('2*sin(0.8x)+0.5*cos(3x)', lambda x: 2 * np.sin(0.8*x) + 0.5 * np.cos(3*x), 'Line 3: Composite Harmonics'),
        ('sqrt(x)*sin(x)', lambda x: np.sqrt(x) * np.sin(x), 'Line 4: Amplitude Modulated')
    ]

    for idx, (expr, base_func, title) in enumerate(line_functions, start=1):
        fig, ax = plt.subplots(figsize=(8, 5))
        y_data, params = generate_line_data(x_line, base_func, n_curves=5)
        
        for i, y in enumerate(y_data):
            ax.plot(x_line, y, marker='o', markersize=4, linewidth=1.5,
                    color=COLORS_TAB10[i % len(COLORS_TAB10)], label=f'Curve {i+1}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        filename = f'line_{idx}.png'
        save_figure(fig, filename)
        
        metadata_list.append({
            'filename': filename,
            'type': 'line',
            'num_curves': 5,
            'x_range': f'{x_line.min():.2f}-{x_line.max():.2f}',
            'y_range': 'auto',
            'base_function': expr,
            'curve_params': '; '.join(params),
            'random_seed': str(RANDOM_SEED)
        })

    print("Generating custom curve charts...")

    # 图1
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    x1 = np.linspace(0, 180, 1000)
    y1, colors1 = gen_curve_1(x1)
    ax1.plot(x1, y1, color=colors1[0], linewidth=2, label='Curve 1')
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Curve 1: Increasing Frequency')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    filename1 = 'curve_1.png'
    save_figure(fig1, filename1)
    metadata_list.append({
        'filename': filename1,
        'type': 'curve',
        'num_curves': 1,
        'x_range': '0-180',
        'y_range': '0-100',
        'base_function': 'piecewise sine with increasing frequency',
        'curve_params': 'black, piecewise parameters',
        'random_seed': str(RANDOM_SEED)
    })

    # 图2
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x2 = np.linspace(-180, 180, 500)
    y2_list, colors2, params2 = gen_curve_2(x2)
    for i, (y, c) in enumerate(zip(y2_list, colors2)):
        ax2.plot(x2, y, color=c, linewidth=2, label=f'Curve {i+1}')
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-48, 96)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Curve 2: Gentle Oscillations')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    filename2 = 'curve_2.png'
    save_figure(fig2, filename2)
    metadata_list.append({
        'filename': filename2,
        'type': 'curve',
        'num_curves': 4,
        'x_range': '-180-180',
        'y_range': '-48-96',
        'base_function': 'amp*sin(freq*x+phase)+20*cos(0.01*x)',
        'curve_params': '; '.join(params2),
        'random_seed': str(RANDOM_SEED)
    })

    # 图3
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    x3 = np.linspace(-180, 180, 800)
    y3_list, colors3, params3 = gen_curve_3(x3)
    for i, (y, c) in enumerate(zip(y3_list, colors3)):
        ax3.plot(x3, y, color=c, linewidth=1.8, label=f'Curve {i+1}')
    ax3.set_xlim(-180, 180)
    ax3.set_ylim(-48, 96)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Curve 3: High Frequency & Large Amplitude')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    filename3 = 'curve_3.png'
    save_figure(fig3, filename3)
    metadata_list.append({
        'filename': filename3,
        'type': 'curve',
        'num_curves': 4,
        'x_range': '-180-180',
        'y_range': '-48-96',
        'base_function': 'amp*sin(freq*x+phase)*cos(0.015*x)+20*sin(0.2*x)',
        'curve_params': '; '.join(params3),
        'random_seed': str(RANDOM_SEED)
    })

    # 图4
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    x4 = np.linspace(87.254, 6174.815, 1000)
    y4_list, colors4, params4 = gen_curve_4(x4)
    for i, (y, c) in enumerate(zip(y4_list, colors4)):
        ax4.plot(x4, y, color=c, linewidth=1.5, label=f'Curve {i+1}')
        if i == 3:
            n_points = 30
            idx = np.random.choice(len(x4), size=n_points, replace=False)
            idx = np.sort(idx)
            x_points = x4[idx]
            y_points = y[idx]
            ax4.scatter(x_points, y_points, facecolors='none',
                        edgecolors=c, s=40, linewidths=1.5)
    ax4.set_xlim(87.254, 6174.815)
    ax4.set_ylim(-4871, 7456)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Curve 4: Eight Curves with Hollow Markers on One')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)
    filename4 = 'curve_4.png'
    save_figure(fig4, filename4)
    metadata_list.append({
        'filename': filename4,
        'type': 'curve',
        'num_curves': 8,
        'x_range': '87.254-6174.815',
        'y_range': '-4871-7456',
        'base_function': 'amp*sin(freq*x+phase)+1000*cos(0.001*x)',
        'curve_params': '; '.join(params4),
        'random_seed': str(RANDOM_SEED)
    })

    # 写入 CSV 元数据
    csv_path = os.path.join(SAVE_DIR, 'figures_metadata.csv')
    fieldnames = ['filename', 'type', 'num_curves', 'x_range', 'y_range',
                  'base_function', 'curve_params', 'random_seed']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)
    print(f"\nAll images saved to {SAVE_DIR}")
    print(f"Metadata saved to {csv_path}")

# -------------------- check 模式 --------------------
def compute_errors(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    maxe = np.max(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return rmse, mae, maxe, r2

def check_mode():
    metadata_path = os.path.join(SAVE_DIR, 'figures_metadata.csv')
    if not os.path.exists(metadata_path):
        print("Error: figures_metadata.csv not found. Run create mode first.")
        return
    
    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['filename']] = row

    corrected_files = glob(os.path.join(SAVE_DIR, '*_corrected*.csv'))
    if not corrected_files:
        print("No *_corrected*.csv files found in ./temp/")
        return

    for csv_file in corrected_files:
        base = os.path.basename(csv_file)
        # 解析文件名：curve_1_corrected4.csv -> orig_name = curve_1.png, curve_index = 4
        # 或者 curve_1_corrected.csv -> curve_index = 1
        match = re.match(r'(line_\d+|curve_\d+)_corrected(\d*)\.csv', base)
        if not match:
            print(f"Unrecognized file pattern: {base}, skipping.")
            continue
        
        orig_stem = match.group(1)
        orig_name = f"{orig_stem}.png"
        curve_index_str = match.group(2)
        curve_index = int(curve_index_str) if curve_index_str else 1  # 默认为第1条
        
        if orig_name not in metadata:
            print(f"Warning: {orig_name} not found in metadata, skipping.")
            continue
        
        meta = metadata[orig_name]
        print(f"\n--- Checking {orig_name} (curve {curve_index}) against {base} ---")
        
        chart_type = meta['type']
        x_range_str = meta['x_range']
        x_min, x_max = map(float, x_range_str.split('-'))
        
        # 读取 CSV 数据（假设无表头，第一列 x，第二列 y）
        data = np.loadtxt(csv_file, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        x_csv = data[:, 0]
        y_csv = data[:, 1]
        
        # 根据类型重建理论曲线
        if chart_type == 'line':
            # 解析基础函数表达式
            expr = meta['base_function']
            # 简单映射到函数（这里手动定义，因为表达式有限）
            if expr == 'sin(x)':
                base_func = lambda x: np.sin(x)
            elif expr == 'cos(x)*exp(-x/10)':
                base_func = lambda x: np.cos(x) * np.exp(-x/10)
            elif expr == '2*sin(0.8x)+0.5*cos(3x)':
                base_func = lambda x: 2 * np.sin(0.8*x) + 0.5 * np.cos(3*x)
            elif expr == 'sqrt(x)*sin(x)':
                base_func = lambda x: np.sqrt(x) * np.sin(x)
            else:
                print(f"Unknown line function: {expr}")
                continue
            
            # 解析曲线参数
            params_str = meta['curve_params']
            # 格式如 "phase=0.00, amp=0.800; phase=0.80, amp=1.139; ..."
            param_parts = params_str.split('; ')
            if curve_index > len(param_parts):
                print(f"Curve index {curve_index} exceeds number of curves {len(param_parts)}")
                continue
            part = param_parts[curve_index-1]
            # 提取 phase 和 amp
            phase_str = part.split(',')[0].split('=')[1]
            amp_str = part.split(',')[1].split('=')[1]
            phase = float(phase_str)
            amp = float(amp_str)
            
            x_theory = np.linspace(x_min, x_max, 500)
            y_theory = amp * base_func(x_theory + phase)
            colors = [COLORS_TAB10[(curve_index-1) % len(COLORS_TAB10)]]
            y_theory_list = [y_theory]
            
        elif chart_type == 'curve':
            # 根据文件名判断使用哪个生成函数
            if orig_stem == 'curve_1':
                x_theory = np.linspace(x_min, x_max, 1000)
                y_array, colors_arr = gen_curve_1(x_theory)  # 返回的是单个数组
                # curve_1 只有一条曲线，curve_index 必须是 1，直接包装成列表
                if curve_index != 1:
                    print(f"Curve index {curve_index} invalid for {orig_name} (only 1 curve)")
                    continue
                y_theory_list = [y_array]      # 包装成列表，使 y_theory_list[0] 是完整数组
                colors = [colors_arr[0]]
            elif orig_stem == 'curve_2':
                x_theory = np.linspace(x_min, x_max, 500)
                y_list, colors, _ = gen_curve_2(x_theory)
                if curve_index > len(y_list):
                    print(f"Curve index {curve_index} out of range")
                    continue
                y_theory_list = [y_list[curve_index-1]]
                colors = [colors[curve_index-1]]
            elif orig_stem == 'curve_3':
                x_theory = np.linspace(x_min, x_max, 800)
                y_list, colors, _ = gen_curve_3(x_theory)
                if curve_index > len(y_list):
                    print(f"Curve index {curve_index} out of range")
                    continue
                y_theory_list = [y_list[curve_index-1]]
                colors = [colors[curve_index-1]]
            elif orig_stem == 'curve_4':
                x_theory = np.linspace(x_min, x_max, 1000)
                y_list, colors, _ = gen_curve_4(x_theory)
                if curve_index > len(y_list):
                    print(f"Curve index {curve_index} out of range")
                    continue
                y_theory_list = [y_list[curve_index-1]]
                colors = [colors[curve_index-1]]
            else:
                print(f"Unknown curve: {orig_stem}")
                continue
        else:
            print(f"Unknown chart type: {chart_type}")
            continue
        
        # 绘制对比图
        fig, ax = plt.subplots(figsize=(8, 5))
        # 理论曲线（虚线）
        ax.plot(x_theory, y_theory_list[0], '--', color=colors[0], linewidth=2,
                label=f'Original Curve {curve_index}')
        # CSV 数据线（实线）
        ax.plot(x_csv, y_csv, '-', color='black', linewidth=2, label='CSV data')
        ax.scatter(x_csv, y_csv, color='black', s=15, alpha=0.7)
        
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Check: {orig_name} Curve {curve_index} vs {base}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 插值计算误差
        y_th_interp = np.interp(x_csv, x_theory, y_theory_list[0])
        rmse, mae, maxe, r2 = compute_errors(y_th_interp, y_csv)
        
        print(f"Error statistics (on CSV x-points):")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE : {mae:.6f}")
        print(f"  Max Error: {maxe:.6f}")
        print(f"  R²  : {r2:.6f}")
        
        plt.show()

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate or check curve/line plots.')
    parser.add_argument('--mode', choices=['create', 'check'], default='create',
                        help='Operation mode: create (default) generates plots, check compares with _corrected*.csv files.')
    args = parser.parse_args()

    if args.mode == 'create':
        create_figures()
    elif args.mode == 'check':
        check_mode()
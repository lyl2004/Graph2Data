#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SciDataRestorer Data Processor
纯算法模块：负责重归一化、角点校正、离群点清洗和数学拟合。
"""

import csv
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 可选的科学计算依赖
try:
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d, PchipInterpolator, splrep, splev
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def map_coords(norm_arr, vmin, vmax, scale):
    if scale == 'linear':
        return vmin + norm_arr * (vmax - vmin)
    elif scale == 'log10':
        if vmin <= 0 or vmax <= 0:
            raise ValueError("对数轴的边界必须大于0")
        log_min, log_max = np.log10(vmin), np.log10(vmax)
        return 10 ** (log_min + norm_arr * (log_max - log_min))
    elif scale == 'inverse':
        if vmin == 0 or vmax == 0:
            raise ValueError("倒数轴边界不能为0")
        inv_min, inv_max = 1/vmin, 1/vmax
        return 1 / (inv_min + norm_arr * (inv_max - inv_min))
    raise ValueError(f"未知的坐标轴类型: {scale}")

def apply_affine_transform(points, src_pts, dst_pts):
    A = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    M, _, _, _ = np.linalg.lstsq(A, dst_pts, rcond=None)
    transformed = np.hstack([points, np.ones((len(points), 1))]) @ M
    return transformed

def corner_calibration(x_norm, y_norm, tolerance_ratio=0.05):
    points = np.column_stack([x_norm, y_norm])
    min_x, max_x = np.min(x_norm), np.max(x_norm)
    min_y, max_y = np.min(y_norm), np.max(y_norm)
    bb_corners = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]
    bb_corners_std = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    diag = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    abs_tolerance = diag * tolerance_ratio
    
    corner_indices, actual_corners = [], []
    for bx, by in bb_corners:
        dist = np.sqrt((x_norm - bx)**2 + (y_norm - by)**2)
        idx = np.argmin(dist)
        if dist[idx] > abs_tolerance:
            return x_norm, y_norm, False, 0, np.ones(len(x_norm), dtype=bool)
        corner_indices.append(idx)
        actual_corners.append([x_norm[idx], y_norm[idx]])
        
    if len(set(corner_indices)) < 4:
        return x_norm, y_norm, False, 0, np.ones(len(x_norm), dtype=bool)
        
    transformed_all = apply_affine_transform(points, np.array(actual_corners), bb_corners_std)
    mask = np.ones(len(points), dtype=bool)
    mask[corner_indices] = False
    
    return transformed_all[mask, 0], transformed_all[mask, 1], True, 4, mask

def clean_mad(x, y, threshold=3.0):
    median_y = np.median(y)
    mad = np.median(np.abs(y - median_y))
    if mad == 0: return x, y, 0
    z_scores = 0.6745 * (y - median_y) / mad
    mask = np.abs(z_scores) <= threshold
    return x[mask], y[mask], np.sum(~mask)

def clean_savgol(x, y, threshold_sigma=3.0):
    if not HAS_SCIPY or len(x) < 5: return x, y, 0
    window = min(11, len(x) if len(x) % 2 != 0 else len(x) - 1)
    if window < 3: return x, y, 0
    y_smooth = savgol_filter(y, window_length=window, polyorder=2)
    residuals = np.abs(y - y_smooth)
    std_res = np.std(residuals)
    if std_res == 0: return x, y, 0
    mask = residuals <= threshold_sigma * std_res
    return x[mask], y[mask], np.sum(~mask)

def fit_data(x, y, fit_method):
    if not HAS_SCIPY and fit_method in ['interp', 'pchip', 'bspline']:
        raise RuntimeError("需要安装 scipy")
        
    x_plot = np.linspace(x.min(), x.max(), max(500, len(x)*2))
    y_std = None
    
    if fit_method == 'interp':
        y_plot = interp1d(x, y, kind='linear')(x_plot)
    elif fit_method == 'pchip':
        y_plot = PchipInterpolator(x, y)(x_plot)
    elif fit_method == 'bspline':
        tck = splrep(x, y, s=len(x)*np.var(y)*0.1)
        y_plot = splev(x_plot, tck)
    elif fit_method == 'gpr':
        if not HAS_SKLEARN: raise RuntimeError("需要安装 scikit-learn")
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        X_train = x.reshape(-1, 1)
        gpr.fit(X_train, y)
        y_plot, y_std = gpr.predict(x_plot.reshape(-1, 1), return_std=True)
    else:
        x_plot, y_plot = x, y
        
    return x_plot, y_plot, y_std

def process_pipeline(params):
    """
    完整的无状态数据处理流水线
    """
    raw_x, raw_y = [], []
    with open(params['file_path'], 'r', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row: continue
            try:
                raw_x.append(float(row[0]))
                raw_y.append(float(row[1]))
            except: pass
    x_norm = np.array(raw_x)
    y_norm = np.array(raw_y)

    calib_applied, corners_found = False, 0
    if params['use_corner']:
        x_norm, y_norm, calib_applied, corners_found, _ = corner_calibration(
            x_norm, y_norm, tolerance_ratio=params['corner_thresh']
        )

    if not calib_applied:
        if params['dirty_mode'] == 'clip':
            x_norm, y_norm = np.clip(x_norm, 0, 1), np.clip(y_norm, 0, 1)
        elif params['dirty_mode'] == 'renorm':
            x_min, x_max = x_norm.min(), x_norm.max()
            y_min, y_max = y_norm.min(), y_norm.max()
            x_norm = (x_norm - x_min) / (x_max - x_min + 1e-10)
            y_norm = (y_norm - y_min) / (y_max - y_min + 1e-10)

    x_act = map_coords(x_norm, params['x_min'], params['x_max'], params['x_scale'])
    y_act = map_coords(y_norm, params['y_min'], params['y_max'], params['y_scale'])

    sort_idx = np.argsort(x_act)
    x_act, y_act = x_act[sort_idx], y_act[sort_idx]

    removed_cnt = 0
    if params['clean'] == 'mad':
        x_act, y_act, removed_cnt = clean_mad(x_act, y_act, params['clean_thresh'])
    elif params['clean'] == 'savgol':
        x_act, y_act, removed_cnt = clean_savgol(x_act, y_act, params['clean_thresh'])

    x_plot, y_plot, y_std = x_act, y_act, None
    if params['fit'] != 'none':
        x_plot, y_plot, y_std = fit_data(x_act, y_act, params['fit'])

    # 返回给 GUI 的结果包
    return {
        'x_act': x_act, 'y_act': y_act,
        'x_plot': x_plot, 'y_plot': y_plot, 'y_std': y_std,
        'removed': removed_cnt, 'calib_applied': calib_applied,
        'corners': corners_found
    }
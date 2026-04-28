#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import tempfile
import numpy as np
import plotly.graph_objects as go
from nicegui import ui, run, app
from csv_processor import process_pipeline

# 常量映射字典
SCALE_OPT = {'线性': 'linear', '对数 (log10)': 'log10', '倒数': 'inverse'}
DIRTY_OPT = {'允许外推': 'extrapolate', '裁剪至边界': 'clip', '重新归一化': 'renorm'}
CLEAN_OPT = {'不剔除': 'none', 'MAD': 'mad', 'Savitzky-Golay': 'savgol'}
FIT_OPT = {'不拟合': 'none', '线性插值': 'interp', 'PCHIP保形': 'pchip', 'B-样条平滑': 'bspline', '高斯过程回归': 'gpr'}

class AppState:
    """会话状态管理：消除全局变量污染，实现双向绑定"""
    def __init__(self):
        self.file_path = None
        self.file_name = "未命名.csv"
        self.out_path = None
        self.has_result = False
        
        # 👇 添加这一行，提前声明加载状态变量，防止 bind 报错
        self.is_processing = False 
        
        # UI 绑定变量
        self.x_min, self.x_max = 0.0, 100.0
        self.y_min, self.y_max = 0.0, 100.0
        self.x_scale, self.y_scale = '线性', '线性'
        self.dirty_mode = '允许外推'
        
        self.use_corner = False
        self.corner_thresh = 0.05
        
        self.clean_method = '不剔除'
        self.clean_thresh = 3.0
        self.fit_method = '不拟合'
        
        self.save_dir = os.path.expanduser("~/Downloads")
        self.line_id = ''

@ui.page('/')
def main_page():
    # 实例化当前用户的会话状态
    state = AppState()

    # --- 顶栏 ---
    with ui.header(elevated=True).classes('items-center justify-between px-4'):
        ui.label('📈 SciDataRestorer Pro').classes('text-h5 font-bold text-white')
        with ui.row():
            ui.button('▶ 运行处理', icon='play_arrow', on_click=lambda: execute_pipeline(state)).props('color=primary')

    # --- 主体布局 ---
    with ui.splitter(value=35).classes('w-full h-full') as splitter:
        # 左侧控制面板
        with splitter.before:
            with ui.column().classes('p-4 gap-3 w-full'):
                
                # 1. 文件上传
                upload = ui.upload(label='选择原始 CSV 数据', auto_upload=True, on_upload=lambda e: handle_upload(e, state))
                upload.props('accept=.csv').classes('w-full')

                # 2. 坐标与范围 (使用 Expansion 节省空间)
                with ui.expansion('📐 物理坐标系设置', icon='straighten').classes('w-full border rounded-lg bg-gray-50').props('default-opened'):
                    with ui.grid(columns=2).classes('w-full gap-2 mt-2'):
                        ui.number('X Min', format='%.2f').bind_value(state, 'x_min').props('dense outlined')
                        ui.number('X Max', format='%.2f').bind_value(state, 'x_max').props('dense outlined')
                        ui.number('Y Min', format='%.2f').bind_value(state, 'y_min').props('dense outlined')
                        ui.number('Y Max', format='%.2f').bind_value(state, 'y_max').props('dense outlined')
                        
                    ui.select(list(SCALE_OPT.keys()), label='X 轴尺度').bind_value(state, 'x_scale').classes('w-full mt-2').props('dense')
                    ui.select(list(SCALE_OPT.keys()), label='Y 轴尺度').bind_value(state, 'y_scale').classes('w-full').props('dense')
                    ui.select(list(DIRTY_OPT.keys()), label='无角点越界处理').bind_value(state, 'dirty_mode').classes('w-full').props('dense')

                # 3. 角点校正
                with ui.expansion('🎯 角点仿射校正', icon='crop').classes('w-full border rounded-lg bg-gray-50'):
                    ui.checkbox('启用四角自适应提取', value=state.use_corner).bind_value(state, 'use_corner')
                    corner_slider = ui.slider(min=0.01, max=0.2, step=0.01).bind_value(state, 'corner_thresh')
                    # 联动：只有勾选了启用，滑块才可用
                    corner_slider.bind_enabled_from(state, 'use_corner')
                    ui.label().bind_text_from(state, 'corner_thresh', backward=lambda v: f'边缘空虚容差: {v:.2f}').classes('text-xs text-gray-500')

                # 4. 清洗与拟合
                with ui.expansion('✨ 算法处理引擎', icon='auto_fix_high').classes('w-full border rounded-lg bg-gray-50'):
                    clean_sel = ui.select(list(CLEAN_OPT.keys()), label='离群点剔除').bind_value(state, 'clean_method').classes('w-full').props('dense')
                    clean_sl = ui.slider(min=1.0, max=10.0, step=0.1).bind_value(state, 'clean_thresh')
                    # 联动：如果不剔除，滑块置灰
                    clean_sl.bind_enabled_from(state, 'clean_method', backward=lambda v: v != '不剔除')
                    ui.label().bind_text_from(state, 'clean_thresh', backward=lambda v: f'剔除强度(Sigma/MAD): {v:.1f}').classes('text-xs text-gray-500')
                    
                    ui.select(list(FIT_OPT.keys()), label='曲线数学拟合').bind_value(state, 'fit_method').classes('w-full mt-2').props('dense')

                # 5. 输出设置
                with ui.expansion('💾 输出与保存', icon='save').classes('w-full border rounded-lg bg-gray-50'):
                    ui.input('输出目录').bind_value(state, 'save_dir').classes('w-full').props('dense')
                    ui.input('文件后缀标识 (如: 4)').bind_value(state, 'line_id').classes('w-full').props('dense')
                    
                    # 联动：有结果时才显示下载按钮
                    download_btn = ui.button('⬇️ 下载当前结果', icon='download', on_click=lambda: ui.download(state.out_path))
                    download_btn.classes('w-full mt-2 text-white').props('color=secondary')
                    download_btn.bind_visibility_from(state, 'has_result')

        # 右侧图表区
        with splitter.after:
            chart = ui.plotly(go.Figure()).classes('w-full h-full p-2')
            
            # 全局加载动画遮罩
            with ui.element('div').classes('absolute inset-0 flex items-center justify-center bg-white/70 z-50').bind_visibility_from(state, 'is_processing', value=False) as overlay:
                ui.spinner('dots', size='xl', color='primary')

    # --- 局部函数定义 ---
# --- 局部函数定义 ---
    async def handle_upload(e, s: AppState):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(await e.file.read())
            s.file_path = tmp.name
            
        # 👇 修改这一行，增加 fallback 兼容不同的 NiceGUI 版本
        s.file_name = getattr(e, 'name', getattr(e, 'filename', 'uploaded.csv'))
        
        ui.notify(f'文件已加载: {s.file_name}', type='positive')

    async def execute_pipeline(s: AppState):
        if not s.file_path:
            ui.notify('请先上传原始 CSV 文件！', type='warning', position='top')
            return

        s.is_processing = True
        try:
            # 提取纯数据参数字典传给后台算法
            params = {
                'file_path': s.file_path,
                'x_min': s.x_min, 'x_max': s.x_max,
                'y_min': s.y_min, 'y_max': s.y_max,
                'x_scale': SCALE_OPT[s.x_scale],
                'y_scale': SCALE_OPT[s.y_scale],
                'dirty_mode': DIRTY_OPT[s.dirty_mode],
                'use_corner': s.use_corner,
                'corner_thresh': s.corner_thresh,
                'clean': CLEAN_OPT[s.clean_method],
                'clean_thresh': s.clean_thresh,
                'fit': FIT_OPT[s.fit_method]
            }

            # 扔进异步线程池执行繁重的矩阵/拟合运算
            res = await run.io_bound(process_pipeline, params)

            # --- 保存结果 ---
            base_name = os.path.splitext(s.file_name)[0]
            suffix = f"_{s.line_id}" if s.line_id else ""
            s.out_path = os.path.join(s.save_dir, f"{base_name}_corrected{suffix}.csv")
            
            with open(s.out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([f"X_{FIT_OPT[s.fit_method]}", f"Y_{FIT_OPT[s.fit_method]}"])
                writer.writerows(zip(res['x_plot'], res['y_plot']))

            # --- 渲染图表 ---
            fig = go.Figure()
            # 原始散点
            fig.add_trace(go.Scatter(x=res['x_act'], y=res['y_act'], mode='markers', name='有效提取点', marker=dict(color='gray', size=5, opacity=0.6)))
            # 拟合线
            if s.fit_method != '不拟合':
                fig.add_trace(go.Scatter(x=res['x_plot'], y=res['y_plot'], mode='lines', name=f'{s.fit_method}', line=dict(color='blue', width=2)))
                if res['y_std'] is not None: # GPR 置信区间
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([res['x_plot'], res['x_plot'][::-1]]),
                        y=np.concatenate([res['y_plot'] - 1.96*res['y_std'], (res['y_plot'] + 1.96*res['y_std'])[::-1]]),
                        fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% 置信'
                    ))
            
            title = f"数据校正分析 | 映射: {s.x_scale}-{s.y_scale} | 剔除: {res['removed']} 点"
            if s.use_corner and res['calib_applied']: title += " | ✅ 四角校正"
            elif s.use_corner: title += " | ⚠️ 角点未找到"
            
            fig.update_layout(title=title, template='plotly_white', hovermode='x unified', margin=dict(l=40, r=20, t=60, b=40))
            if s.x_scale == '对数 (log10)': fig.update_xaxes(type="log")
            if s.y_scale == '对数 (log10)': fig.update_yaxes(type="log")
            
            chart.update_figure(fig)
            s.has_result = True
            
            # --- 动态反馈 ---
            msg = f"处理完成！已保存至: {s.out_path}"
            ui.notify(msg, type='positive' if res['calib_applied'] or not s.use_corner else 'warning', position='bottom-right')

        except Exception as ex:
            ui.notify(f"算法执行出错: {str(ex)}", type='negative', position='top')
        finally:
            s.is_processing = False

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='SciDataRestorer', favicon='📈', reload=False, dark=False)
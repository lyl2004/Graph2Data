# Graph2Data

Graph2Data 是一个面向科学图表的数据恢复项目，目标是从论文图、实验图或软件导出的曲线图中恢复可计算的数据表。

当前项目包含两条工作线：

1. 已成型的 CSV 校正工具：接收 Automeris / WebPlotDigitizer 等工具导出的曲线点 CSV，完成坐标映射、清洗、拟合和导出。
2. 尚在原型阶段的本地图像提取工具：尝试直接从图像中检测坐标轴、识别文字、提取曲线特征、补全曲线并还原数值。

后续工作的核心方向是逐步补完本地图像提取链路，使其最终替代网站工具导出的中间 CSV，同时保留 CSV 校正模块作为统一后处理和质量控制内核。

## 当前阶段快照

当前项目已经从 `tests/` 下的松散实验脚本，推进到 `src/graph2data/` 下的第一版结构化图像提取包。现阶段重点不是 GUI 完整化，而是先把“图像 -> 曲线 mask -> 有序路径 -> 质量评估”的核心链路做成可复现、可量化、可持续迭代的工程基线。

已经形成闭环的能力：

- 合成 benchmark 生成：输出图像、坐标轴真值、曲线真值数据、每条曲线真值 mask。
- 颜色原型提取：支持彩色曲线、黑色曲线、灰色曲线和基础抗锯齿容差。
- 曲线 mask 生成：根据曲线颜色原型生成单曲线二值 mask，并支持图例区域排除。
- Mask 清理：会过滤明显边框/网格残留，并保守移除紧贴边缘的极小刻度残片，避免破坏虚线和点线端点。
- 骨架化与路径追踪：将 mask 转为中心线 skeleton，再生成有序像素路径。
- 通用路径重建：当普通 `curve_path` 只覆盖少量 skeleton 像素且存在明显 junction 干扰时，`lines.py` 现可按整条 skeleton 的平滑 x 投影重建更完整的中心线，并在 warnings 中标记 `path_rebuilt_from_skeleton_x_projection`。
- 断裂补全记录：对虚线/点线等多连通片段执行基础 gap linking，并保存补全区间和点级置信度。
- 图例污染回归测试：支持绘图区内图例合成场景，能量化排除图例前后的误差变化。
- 图例 item 诊断解析：已能把检测到的 legend bbox 分块为若干 item，提取样本区域、文本区域、样本颜色和粗略线型，并输出 `debug/legend_items.png`。
- 图例标签传播：开启 OCR 时，会把落在 legend item 文本区域内的 OCR 文本赋给 `LegendItem.label`，并继续传播到 `CurveVisualPrototype`、prototype-bound path、prototype-bound data series 和导出的 CSV `label` 列；OCR 缺失或识别失败不会阻断图形学 pipeline。
- 图例 item 行数修正：synthetic 图例底部残留行已在 row 分块阶段过滤；`same_color_marker_curves` 的 legend item 数现与 4 条 truth 曲线对齐，`same_gray_linestyle_curves` 的 legend item 数现与 6 条 truth 曲线对齐。
- 同灰线型重大修正：`line_style_instance` 当前已加入局部 x-neighborhood rank 分组、组件级 skeleton 主路径拼接和实例内 y 离群组件裁剪；prototype-bound path 的 RMSE / coverage 已进入稳定区间。
- 同灰线型连续化节点：prototype-bound line-style path 当前会用方向、y 跳变和局部切线约束筛选可连接的 dash/dot gap，执行低置信度 gap interpolation；补全点写入 `completed_ranges` 并以较低点级置信度导出，debug overlay 会用不同颜色显示低置信度补全段。
- 下一阶段 synthetic 场景：已支持 `marker_curves`、`line_marker_curves`、`crossing_line_marker_curves`、`same_color_line_marker_curves`、`same_gray_line_marker_curves`、`crossing_same_gray_line_marker_curves`、`same_color_marker_curves`、`same_gray_linestyle_curves`、`dense_legend_curves` 生成，用于后续 marker/线型分离和图例绑定实验。
- 组件分类诊断：已能把每条 mask 的 skeleton 连通组件分类为 `line_like`、`marker_like` 或 `noise`，输出结构化 `line_components`、质量报告 `component_summary` 和 `debug/component_classification.png`。
- Marker 候选诊断：已能从原始 mask 连通组件中提取 marker 候选中心点、bbox、填充率、圆度和粗略形状，输出结构化 `marker_candidates`、质量报告 `marker_summary` 和 `debug/marker_candidates.png`。
- 线+marker 局部分离：marker 候选检测已加入距离变换局部 blob 检测，可从与线条相连的 marker 中恢复候选中心点；`same_color_marker_curves` 的 marker candidate recall 已从 0 提升到约 0.94。
- 同色 marker 分组诊断：prototype-binding benchmark 已加入局部 x-neighborhood rank 分组，可把同色 marker 候选按轨迹分成多组；`same_color_marker_curves` 的 marker group assignment accuracy 当前约 1.0。
- Marker 曲线实例输出：pipeline 已输出诊断级 `marker_curve_instances`，并生成 `debug/marker_curve_instances.png`；当前实例默认基于 marker 中心分组，在 line-like 组件明显增多的场景下可切换到 trajectory-ransac 分组尝试维持 crossing 曲线身份；已加入保守过滤，避免 `same_gray_linestyle_curves` 中的虚线/点线碎片被当成 marker 实例抢走绑定。
- 同灰线型实例诊断：已能把 `line_like` 组件按垂直轨迹分组为诊断级 `line_style_curve_instances`，并输出 `debug/line_style_curve_instances.png`；在 `same_gray_linestyle_curves` 中当前可作为图例顺序约束的实例级绑定目标，并可生成 prototype-bound path/data。
- Prototype 绑定诊断：已能对 legend item 生成的 `CurveVisualPrototype`、绘图区曲线结果、`marker_curve_instances` 和 `line_style_curve_instances` 计算诊断级绑定评分，输出 `prototype_bindings` 和质量报告 `binding_summary`；在 `same_color_marker_curves`、`same_gray_line_marker_curves`、`crossing_same_gray_line_marker_curves` 与 `same_gray_linestyle_curves` 中 binding accuracy 当前约 1.0。
- Prototype-bound path/data 输出：当最佳绑定目标是 `curve`、`marker_instance` 或 `line_style_instance` 时，pipeline 都会生成 `prototype_bound_paths`；其中 direct curve 绑定会复用原始曲线路径，marker 绑定在可唯一回溯到源曲线时会优先复用连续线条 path，而不是只保留稀疏 marker 中心点。开启 `--map_data` 且提供坐标范围时，会额外输出 `data/prototype_bound_curves.csv` 和 `prototype_bound_data_series`，其中会保留图例标签字段。
- 路径质量诊断：quality report 的 `path_summary` 与逐曲线 `curves[]` 现已增加 `mean_path_coverage_ratio` / `path_coverage_ratio` 和 `rebuilt_path_count` / `path_rebuilt`，可直接观察普通 path 是否因 marker junction 触发了重建。
- 质量评估：包含路径级 Chamfer/Hausdorff/truth-to-pred 指标，以及 mask 级 IoU/F1 和 2px 容差 F1。

当前固定 suite 的参考结果：

```text
basic_curves:
  mean_chamfer_distance_px ≈ 0.85
  mean_mask_tolerant_f1    ≈ 0.89

achromatic_curves:
  mean_chamfer_distance_px ≈ 0.87
  mean_mask_tolerant_f1    ≈ 0.89

local_occlusion_curves:
  mean_chamfer_distance_px ≈ 0.97
  mean_hausdorff_distance_px ≈ 14.2
  mean_data_y_rmse ≈ 0.012

crossing_curves:
  mean_chamfer_distance_px ≈ 0.95
  mean_hausdorff_distance_px ≈ 9.7
  mean_data_y_rmse ≈ 0.0095

legend_inside_curves，不排除图例:
  mean_chamfer_distance_px ≈ 5.20
  mean_hausdorff_distance_px ≈ 252.40

legend_inside_curves，排除图像启发式检测到的图例区域:
  mean_chamfer_distance_px ≈ 0.84
  mean_hausdorff_distance_px ≈ 6.40

legend_inside_curves，排除合成图例区域:
  mean_chamfer_distance_px ≈ 0.84
  mean_hausdorff_distance_px ≈ 6.40
```

阶段判断：

```text
已完成：第一版纯图形学分辨基线和 benchmark 闭环。
正在推进：图像启发式图例检测的真实图扩展、坐标轴和 OCR 结果的稳定融合、困难曲线场景扩展。
尚未完成：刻度 OCR 语义解析、复杂图例语义解析、复杂交叉/重叠曲线归属、CSV 校正模块作为 pipeline 后处理内核的正式复用。
```

## 当前节点展示

截至当前节点，项目已经具备一次阶段展示所需的最小完整材料：

- 一条正式 pipeline：`图像 -> 坐标轴 -> 曲线颜色 -> mask -> path -> data CSV -> quality/report.json`。
- 五类固定 synthetic 场景：基础彩色曲线、黑灰/彩色混合曲线、局部短遮挡、轻度交叉、绘图区内图例污染。
- 一组自动质量门：pytest 回归测试、compileall、固定 benchmark suite。
- 一组可视化 artifact：mask、overview、paths_overlay、逐曲线 path overlay、数据 CSV、质量报告。
- `paths_overlay.png` 会跳过不连续片段之间的超长连接，避免把尚未可靠连接的 mask 污染片段画成跨图直线。

推荐展示命令分为四组。

第一组：验证工程质量门。

```powershell
$env:PYTHONPATH='src'
pixi run python -m pytest -q
pixi run python -m compileall -q src tests
```

第二组：运行完整 synthetic benchmark suite。该 suite 会生成并评估 5 类场景：

```text
basic_curves             基础彩色曲线
achromatic_curves        黑灰/彩色混合曲线
local_occlusion_curves   局部短遮挡/断裂
crossing_curves          颜色可分的轻度交叉曲线
legend_inside_curves     绘图区内图例污染
```

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.benchmark --suite --suite_out temp\suite_stage_demo
Get-Content temp\suite_stage_demo\results\suite_summary.json
```

第三组：运行真实样例图的完整 pipeline，输出 mask、path、数据 CSV、debug 图和质量报告。

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\stage_demo_pipeline.json --map_data --x_min 0 --x_max 100 --y_min -10 --y_max 10 --artifact_dir temp\stage_demo_artifacts --debug_artifacts
Get-ChildItem -Recurse temp\stage_demo_artifacts
Get-Content temp\stage_demo_artifacts\quality\report.json
```

第四组：单独复现重点困难场景，便于展示局部能力。

```powershell
$env:PYTHONPATH='src'

# 局部短遮挡/断裂
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name local_occlusion_curves --palette basic --local_occlusion
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\local_occlusion_curves --mode predicted-mask --mask_out temp\local_occlusion_predicted_masks --out temp\local_occlusion_pred_benchmark.json
Get-Content temp\local_occlusion_pred_benchmark.json

# 轻度交叉曲线
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name crossing_curves --palette basic --crossing_curves
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\crossing_curves --mode predicted-mask --mask_out temp\crossing_predicted_masks --out temp\crossing_pred_benchmark.json
Get-Content temp\crossing_pred_benchmark.json

# 绘图区内图例污染，对比不排除图例和图像启发式排除图例
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name legend_inside_curves --palette basic --legend_inside
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\legend_inside_curves --mode predicted-mask --mask_out temp\legend_no_exclude_masks --out temp\legend_no_exclude.json
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\legend_inside_curves --mode predicted-mask --detect_legend --mask_out temp\legend_detected_exclude_masks --out temp\legend_detected_exclude.json
Get-Content temp\legend_no_exclude.json
Get-Content temp\legend_detected_exclude.json
```

展示时可以重点说明：

- 目前对颜色可分、函数型曲线的恢复已经形成稳定工程闭环。
- 局部短遮挡和轻度交叉已经进入固定回归测试，数据空间误差仍保持在较低水平。
- 绘图区内图例污染会显著拉高误差，但图像启发式图例排除可以把指标恢复到正常范围。
- `tests/test1.png` 属于真实灰度样例；当前已能检测并排除右上角无框图例，`paths_overlay.png` 也不再绘制跨片段长连线。但 mask 中仍会把同色/近灰度线条、marker 和部分刻度残片混在一起。当前展示应把它作为诊断样例，而不是证明真实灰度图已经完全解决。
- 可用 `--line_filter_marker_like` 做实验性 marker 过滤对比；该开关会在同一 mask 内存在较长线段组件时跳过紧凑 marker-like 组件，但目前不默认开启，因为它在虚线/点线场景中仍需更强判别条件。
- CSV 输出已包含 `label` 列；只有开启 OCR 且文本框能匹配到 legend item 文本区域时，该列才会有真实图例文本，否则保持为空并继续输出数值结果。
- 当前尚不是“任意论文图全自动解析”，下一阶段应优先推进图例样本解析、marker/线型实例分离、刻度 OCR 语义解析和复杂灰度图归属。

展示时重点查看的输出文件：

```text
temp/suite_stage_demo/results/suite_summary.json
temp/stage_demo_pipeline.json
temp/stage_demo_artifacts/data/curves.csv
temp/stage_demo_artifacts/data/prototype_bound_curves.csv
temp/stage_demo_artifacts/debug/overview.png
temp/stage_demo_artifacts/debug/legend_items.png
temp/stage_demo_artifacts/debug/component_classification.png
temp/stage_demo_artifacts/debug/marker_candidates.png
temp/stage_demo_artifacts/debug/marker_curve_instances.png
temp/stage_demo_artifacts/debug/line_style_curve_instances.png
temp/stage_demo_artifacts/debug/prototype_bound_paths.png
temp/stage_demo_artifacts/debug/paths_overlay.png
temp/stage_demo_artifacts/debug/paths/*.png
temp/stage_demo_artifacts/masks/*.png
temp/stage_demo_artifacts/quality/report.json
```

## 下一阶段工作计划

当前节点已经证明基础 pipeline、benchmark 和 debug artifact 可以稳定运行。下一阶段应从“能跑通”转向“能区分真实图中的不同视觉实例”，核心目标是解决真实灰度图和论文截图中最常见的曲线归属问题。

下一阶段名称：图例样本解析与曲线实例分离。

阶段目标：

```text
把当前按颜色/灰度原型生成 mask 的流程，升级为“图例样本 -> 曲线视觉特征 -> 实例级 mask/path”的流程。
重点解决同色或近灰度曲线、marker 曲线、虚线/点线、图例样本和绘图区曲线之间的绑定问题。
```

优先级最高的工作主线：

1. 图例样本解析
   - 已在 `legend.py` 中加入诊断级 legend item 分块，可把已检测到的 legend bbox 分解为若干 legend item。
   - 已对每个 item 提取样本区域、文本区域、颜色/灰度和粗略线型；开启 OCR 时已可按文本区域匹配并传播标签文本，marker 形状和复杂 OCR 语义仍待增强。
   - 已新增 `LegendItem` 和 `CurveVisualPrototype` 结构，用于描述一条曲线的视觉身份。
   - 已输出 debug artifact：`debug/legend_items.png`，能看到每个图例 item 的 bbox、样本区域和文本区域。

2. Marker 与线型分离
   - 已在 `lines.py` 中把 skeleton 组件分成 `line_like`、`marker_like`、`noise` 三类，并接入 pipeline 结构化输出。
   - 当前 `--line_filter_marker_like` 仍只是实验开关；新增组件分类默认只做诊断，不默认删除任何组件。
   - 已从原始 mask 组件中提取 marker 中心点和形状特征，不只依赖 skeleton 几何。
   - 已加入距离变换局部 blob 检测，能处理 marker 与线条连成同一组件的场景。
   - 当前 marker 候选仍是诊断级结果，尚未用于拆分同色曲线或重写 path。
   - 对 marker 曲线保留 marker 中心点，同时对连续线段保留 centerline，避免 marker 既污染路径又丢失曲线身份。
   - 已增加 synthetic case 生成开关：纯 marker、同色不同 marker、同灰不同线型、密集图例；后续继续补线+marker 联合评测和虚线密度变化指标。

3. 图例到绘图区的实例绑定
   - 已将图例中提取到的颜色、灰度、粗线型、粗 marker 风格用于生成诊断级 binding score。
   - 下一步将 binding score 用于约束绘图区 mask/path，而不是只输出诊断报告。
   - 对彩色图继续使用颜色作为强特征；对灰度图增加线型、marker、局部方向和组件周期作为补充特征。
   - 已建立第一版绑定评分：颜色/灰度相似度、line-like 存在性、marker 候选相似度。
   - 已输出每条 prototype 到曲线结果的绑定分数、置信度和 warnings，避免在歧义场景下静默给出错误结果。

4. Benchmark 扩展与质量门
   - 在 `synthetic.py` 中新增下一阶段固定场景：
     - `marker_curves`          已可生成
     - `line_marker_curves`     已可生成
     - `crossing_line_marker_curves` 已可生成
     - `same_color_line_marker_curves` 已可生成
     - `same_gray_line_marker_curves` 已可生成
     - `crossing_same_gray_line_marker_curves` 已可生成
     - `same_color_marker_curves` 已可生成
     - `same_gray_linestyle_curves` 已可生成
     - `dense_legend_curves`    已可生成
   - 在 `benchmark.py` 中已增加诊断级 legend item / prototype 绑定指标：
     - 图例 item 检出数量
     - `binding_count`
     - `binding_accuracy`
     - `mean_best_score`
     - `truth_marker_count`
     - `marker_candidate_recall`
     - `marker_candidate_precision`
     - `marker_group_assignment_accuracy`
     - `labeled_prototype_bound_path_count`
     - `labeled_prototype_bound_data_count`
   - 后续继续补：
     - 同色/同灰曲线实例分离准确率
   - pytest 中新增小而稳定的单元测试，先覆盖图例 item 分解和 marker/line 组件分类。

5. 真实图诊断闭环
   - 继续使用 `tests/test1.png` 作为真实灰度诊断图。
   - 下一阶段的真实图目标不是一次性完全解析它，而是让 artifact 明确显示：
     - 图例 item 是否被正确分块。
     - marker-like 组件是否被识别为 marker，而不是直接混入路径。
     - 同灰度曲线为何被合并或为何能拆分。
   - 所有真实图改动必须先通过 synthetic benchmark，避免为了一个真实样例破坏基础场景。

建议实现顺序：

```text
1. 已新增 LegendItem / CurveVisualPrototype 数据结构。
2. 已实现 legend bbox 内 item 分块和样本区域/文本区域提取。
3. 已为 legend item 输出 debug overlay 和 JSON。
4. 下一步扩展 synthetic：生成 marker 曲线和同色/同灰曲线。
5. 已实现 marker-like / line-like / noise skeleton 组件分类，不默认删除任何组件。
6. 已提取 marker 中心点与形状特征，并已输出 legend prototype 到 marker_curve_instances 的绑定诊断；下一步用 binding score 约束 mask/path 提取，先在 synthetic 上验收。
7. 最后回到 tests/test1.png 做真实图 A/B 诊断。
```

当前可直接运行的 prototype binding 诊断命令：

```powershell
$env:PYTHONPATH='src'

pixi run python -m graph2data.synthetic --out temp\binding_benchmark --name same_color_marker_curves --same_color_marker_curves --legend_inside --curves 4
pixi run python -m graph2data.benchmark --case temp\binding_benchmark\same_color_marker_curves --mode prototype-binding --out temp\binding_benchmark\same_color_marker_binding.json

pixi run python -m graph2data.synthetic --out temp\binding_benchmark --name line_marker_curves --line_marker_curves --legend_inside --curves 4
pixi run python -m graph2data.benchmark --case temp\binding_benchmark\line_marker_curves --mode prototype-binding --out temp\binding_benchmark\line_marker_binding.json

pixi run python -m graph2data.synthetic --out temp\binding_benchmark --name crossing_line_marker_curves --crossing_line_marker_curves --legend_inside --curves 4
pixi run python -m graph2data.benchmark --case temp\binding_benchmark\crossing_line_marker_curves --mode prototype-binding --out temp\binding_benchmark\crossing_line_marker_binding.json

pixi run python -m graph2data.synthetic --out temp\binding_benchmark --name same_color_line_marker_curves --same_color_line_marker_curves --legend_inside --curves 4
pixi run python -m graph2data.benchmark --case temp\binding_benchmark\same_color_line_marker_curves --mode prototype-binding --out temp\binding_benchmark\same_color_line_marker_binding.json

pixi run python -m graph2data.synthetic --out temp\binding_benchmark --name same_gray_line_marker_curves --same_gray_line_marker_curves --legend_inside --curves 4
pixi run python -m graph2data.benchmark --case temp\binding_benchmark\same_gray_line_marker_curves --mode prototype-binding --out temp\binding_benchmark\same_gray_line_marker_binding.json

pixi run python -m graph2data.synthetic --out temp\binding_benchmark --name same_gray_linestyle_curves --same_gray_linestyle_curves --legend_inside --curves 6
pixi run python -m graph2data.benchmark --case temp\binding_benchmark\same_gray_linestyle_curves --mode prototype-binding --out temp\binding_benchmark\same_gray_binding.json
```

当前 `line_marker_curves` 诊断参考：

```text
普通 path benchmark:
curve_count = 4
valid_curve_count = 4
mean_chamfer_distance_px ≈ 1.36
mean_data_y_rmse ≈ 0.012
mean_data_x_coverage_ratio ≈ 0.97
mean_path_coverage_ratio ≈ 0.93
rebuilt_path_count 有结构化输出

prototype-binding benchmark:
legend_item_count = 4
binding_accuracy ≈ 1.0
prototype_bound_path_count ≈ 4
valid_prototype_bound_path_count ≈ 4
valid_prototype_bound_data_count ≈ 4
labeled_prototype_bound_path_count ≈ 4
mean_prototype_bound_data_y_rmse ≈ 0.010
mean_prototype_bound_data_x_coverage_ratio ≈ 0.99
```

当前 `crossing_line_marker_curves` 诊断参考：

```text
普通 path benchmark:
mean_data_y_rmse ≈ 0.010
mean_data_x_coverage_ratio ≈ 0.99
mean_path_coverage_ratio ≈ 0.93
rebuilt_path_count ≈ 1

prototype-binding benchmark:
binding_accuracy ≈ 1.0
mean_prototype_bound_data_y_rmse ≈ 0.008
mean_prototype_bound_data_x_coverage_ratio ≈ 0.99
```

当前 `same_color_line_marker_curves` 诊断参考：

```text
普通 path benchmark:
mean_data_y_rmse ≈ 0.012
mean_data_x_coverage_ratio ≈ 1.00
mean_path_coverage_ratio ≈ 0.92
rebuilt_path_count ≈ 1

prototype-binding benchmark:
binding_accuracy ≈ 1.0
mean_prototype_bound_data_y_rmse ≈ 0.007
mean_prototype_bound_data_x_coverage_ratio ≈ 0.90
```

当前 `same_gray_line_marker_curves` 诊断参考：

```text
普通 path benchmark:
mean_data_y_rmse ≈ 0.012
mean_data_x_coverage_ratio ≈ 1.00
mean_path_coverage_ratio ≈ 0.92
rebuilt_path_count ≈ 1

prototype-binding benchmark:
binding_accuracy ≈ 1.0
best_target_type 当前优先为 marker_instance
mean_prototype_bound_data_y_rmse ≈ 0.008
mean_prototype_bound_data_x_coverage_ratio ≈ 0.92
```

当前 `crossing_same_gray_line_marker_curves` 诊断参考：

```text
普通 path benchmark:
mean_data_y_rmse ≈ 0.010
mean_data_x_coverage_ratio ≈ 0.99
mean_path_coverage_ratio ≈ 0.93
rebuilt_path_count ≈ 1

prototype-binding benchmark:
binding_accuracy ≈ 1.0
best_target_type 当前稳定为 curve
prototype_bound_path_count = curve_count = 4
mean_prototype_bound_data_y_rmse 当前约 0.21
mean_prototype_bound_data_x_coverage_ratio 当前约 0.99

说明：该场景的 prototype 身份绑定已通过细灰阶 legend guidance、一对一 target 分配、guided-gray 低面积置信策略和 marker-instance 降权收口。
当前主要剩余问题是细灰阶 direct path 的局部污染仍高于普通 truth-mask path，数据 RMSE 尚明显高于普通 path。
```

当前 `same_color_marker_curves` 诊断参考：

```text
truth_marker_count ≈ 72
marker_candidate_count ≈ 68
marker_candidate_recall ≈ 0.94
marker_candidate_precision ≈ 1.0
marker_group_assignment_accuracy ≈ 1.0
legend_item_count = curve_count = 4
prototype binding 已优先绑定到 marker_curve_instances
prototype_bound_path_count ≈ 4
mean_prototype_bound_data_y_rmse ≈ 0.008
mean_prototype_bound_data_x_coverage_ratio ≈ 0.90
prototype_bound_data_series_count ≈ 4  # 需要 --map_data 和坐标范围
binding_accuracy ≈ 1.0
```

当前 `same_gray_linestyle_curves` 诊断参考：

```text
marker_curve_instance_count = 0
line_style_curve_instance_count >= curve_count
prototype binding 已优先绑定到 line_style_curve_instances
prototype_bound_path_count >= curve_count
prototype_bound_data_series_count >= curve_count  # 需要 --map_data 和坐标范围
当前 line-style instance 分组已优先使用局部 x-neighborhood rank，再回退到 y 聚类
line-like 组件当前已保存局部 skeleton 主路径，prototype-bound path 优先拼接真实组件路径点
legend_item_count = curve_count = 6
实例内当前会裁剪极端 y 离群组件，减少边框/残留片段污染
相邻 dash/dot 片段当前会执行低置信度 gap interpolation，补全点保存在 completed_ranges
mean_prototype_bound_data_y_rmse 当前约 0.24 - 0.28
mean_prototype_bound_data_x_coverage_ratio 当前约 0.87 - 0.93
mean_prototype_bound_completed_point_ratio 有结构化输出
prototype_bound_path_summary.completed_point_ratio 有结构化输出
binding_accuracy ≈ 1.0
```

说明：`line_marker_curves`、`crossing_line_marker_curves`、`same_color_line_marker_curves` 和 `same_gray_line_marker_curves` 已用于验证普通 path 质量门、prototype-bound 输出、标签传播和 CSV/data 一致性；`crossing_same_gray_line_marker_curves` 的 prototype binding 身份已收口并稳定走 direct curve，但 prototype-bound 数据误差仍需通过更可靠的细灰阶 mask、局部污染过滤或跨交叉点轨迹保持继续降低。通用 path 提取已能在低覆盖 junction 场景下触发平滑 skeleton x 投影重建。`same_gray_linestyle_curves` 的绑定准确率、legend item 行数、prototype-bound path coverage 和数据误差现在都已收口到可展示区间。gap interpolation 已加入方向、y 跳变和局部切线约束，但仍是线性插值。下一步应继续引入曲率/局部趋势拟合，进一步压低 Hausdorff 并提升 path 平滑度。

下一阶段验收标准：

```text
pytest 全部通过
compileall 通过
原有 fixed benchmark suite 不退化
新增 marker / same-gray / legend-item benchmark 有结构化指标
真实样例 tests/test1.png 至少输出 legend_items.png、component_classification.png 和更清晰的 warnings
README 中更新当前能力边界和展示命令
```

下一阶段非目标：

```text
暂不做 GUI。
暂不引入机器学习作为主路径。
暂不承诺任意论文图全自动解析。
暂不把实验性 marker 过滤设为默认，除非 synthetic benchmark 证明不伤害虚线/点线。
```

## 当前项目结构

```text
Graph2Data/
  assets/              CSV GUI 静态资源、字体和样式文件
  benchmarks/          合成 benchmark 输出目录，生成结果默认不纳入版本控制
  src/
    csv_gui.py         CSV 校正工具 NiceGUI 可视化入口
    csv_processor.py   CSV 坐标映射、清洗、拟合核心逻辑
    csv_create.py      早期合成测试图与理论曲线校验工具
    config.json        CSV 校正参数配置草稿
    tempreadme         CSV 校正模块设计记录
    graph2data/        当前重点推进的正式图像提取包
  tests/
    test_pre.py        早期坐标轴检测原型
    test_ocr.py        早期 OCR 文本识别原型
    test_colors.py     早期曲线颜色提取原型
    test_lines.py      早期曲线线条提取与补全原型
    test_worker.py     早期图像分析组合工作流原型
    test_draw.py       早期合成测试图生成工具
  temp/                本地临时输出、debug artifact 和 benchmark 运行结果
  pixi.toml            Pixi 环境配置
  pixi.lock            锁定依赖版本
```

## 当前重构入口

第一阶段工程化重构已经开始，新的正式包位于：

```text
src/graph2data/
```

当前已建立的模块：

```text
models.py       统一数据结构
image_io.py     图像和 JSON artifact I/O
axes.py         坐标轴和绘图区检测
layout.py       基于绘图区的九宫格划分和 OCR 归类
legend.py       初版图例区域检测
ocr.py          RapidOCR 包装
colors.py       曲线颜色原型提取
masks.py        根据颜色原型生成曲线 mask
lines.py        mask 骨架化和有序路径追踪
mapping.py      像素路径到数据坐标的映射和 CSV 输出
pipeline.py     结构化 pipeline 入口
synthetic.py    合成 benchmark 生成器
quality.py      最小质量评估工具
benchmark.py    批量 benchmark runner
```

可以用以下命令运行结构化 pipeline：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1.json --colors
```

如需在正式 pipeline 中继续生成曲线 mask 和有序像素路径：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1_paths.json --paths --artifact_dir temp\pipeline_test1_artifacts --debug_artifacts
```

其中 `--paths` 会自动启用颜色原型和 mask 提取，`--artifact_dir` 会保存每条曲线的 mask PNG 和 `quality/report.json`。`--debug_artifacts` 会额外输出 overview、全局路径 overlay 和逐曲线路径 overlay，便于定位坐标轴、图例、mask 和 path 的问题。

如需提供坐标范围并导出数据空间 CSV：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1_mapped.json --map_data --x_min 0 --x_max 100 --y_min -10 --y_max 10 --artifact_dir temp\pipeline_test1_mapped_artifacts --debug_artifacts
```

其中 `--map_data` 会自动启用颜色原型、mask、path 和数据坐标映射，并输出 `data/curves.csv` 和 `quality/report.json`。

如需对真实灰度样例试验 marker-like 紧凑组件过滤：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1_marker_filter.json --map_data --x_min 0 --x_max 100 --y_min -10 --y_max 10 --artifact_dir temp\pipeline_test1_marker_filter_artifacts --debug_artifacts --line_filter_marker_like
```

该开关当前是实验能力，适合对比 `quality/report.json` 中的 `marker_like_components_skipped`、`path_point_count` 和 `debug/paths_overlay.png`。它不是默认策略，避免误伤虚线/点线曲线的真实短片段。

如需开启 OCR：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1_ocr.json --ocr --colors
```

当前新 pipeline 的目标不是一次性替代所有 `tests/` 原型，而是先建立稳定的数据接口和 JSON 输出。后续会逐步把 `tests/test_pre.py`、`test_ocr.py`、`test_colors.py`、`test_lines.py` 中成熟的算法迁移进正式模块。

当前下一步工程任务：

1. 继续推进 `legend.py`：当前已完成诊断级“图例 item 分块 + 样本区域/文本区域/颜色/粗线型提取”，并已打通可选 OCR 标签传播；下一步补强复杂图例文本语义和 marker 形状细分。
2. 继续扩展 prototype-bound 输出：`line_marker_curves`、`crossing_line_marker_curves`、`same_color_line_marker_curves` 和 `same_gray_line_marker_curves` 已验证 bound path、标签传播、data series 和 CSV 字段一致性；`crossing_same_gray_line_marker_curves` 已稳定为 direct curve 绑定，下一步继续把该场景 prototype-bound data RMSE 从约 0.21 压向普通 path 的 0.01 量级，并覆盖更复杂遮挡场景。
3. 增强 `lines.py`：marker junction 导致的短路径重建现已前移到通用 path extraction，并加入平滑 x 投影中心线；下一步将该策略升级为更贴合局部曲率的路径拟合。
4. 在 `pipeline.py` 中继续增加真实灰度图的诊断信息，重点解释同灰度曲线被合并或拆开的原因。
5. 保持现有 benchmark 不退化，再逐步接入真实灰度图 `tests/test1.png` 的诊断改进。

生成一个带真值数据的合成 benchmark：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name basic_curves
```

该命令会输出：

```text
benchmarks/synthetic/basic_curves/
  image.png
  manifest.json
  truth_axes.json
  truth_curves.json
  truth_data.csv
  masks/
```

运行 pipeline 并评估坐标轴检测：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img benchmarks\synthetic\basic_curves\image.png --out temp\basic_curves_pipeline.json --colors
pixi run python -m graph2data.quality --prediction temp\basic_curves_pipeline.json --truth_axes benchmarks\synthetic\basic_curves\truth_axes.json
```

从单条曲线 mask 提取有序像素路径：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.lines --mask benchmarks\synthetic\basic_curves\masks\curve_00.png --curve_id curve_00 --out temp\curve_00_path.json
```

评估提取路径与真值曲线的像素误差：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.quality --path temp\curve_00_path.json --truth_axes benchmarks\synthetic\basic_curves\truth_axes.json --truth_data benchmarks\synthetic\basic_curves\truth_data.csv --curve_id curve_00
```

批量评估合成案例的所有曲线 mask：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\basic_curves --out temp\basic_curves_benchmark.json
```

批量评估“颜色原型 -> 预测 mask -> 路径”的图形学分辨链路：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\basic_curves --mode predicted-mask --mask_out temp\basic_predicted_masks --out temp\basic_pred_benchmark.json
```

运行固定 benchmark suite：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.benchmark --suite --suite_out temp\suite_check
```

固定 suite 当前包含三类场景：

```text
basic_curves          彩色多曲线，图例在绘图区外
achromatic_curves     黑色/灰色/彩色曲线共存
local_occlusion_curves 曲线局部短遮挡/断裂，真值数据保持完整
crossing_curves        颜色可分的轻度交叉曲线
legend_inside_curves  图例位于绘图区内，用于测试图例污染
```

运行自动化回归测试：

```powershell
$env:PYTHONPATH='src'
pixi run python -m pytest -q
```

当前 pytest 会检查三类质量门：

- `pipeline --paths/--map_data` 能在真实样例上输出曲线 mask、有序路径、数据序列、CSV 和 debug artifact，且不会把普通曲线区域误检为图例。
- `mapping.py` 能把已知像素路径正确映射到数据坐标。
- 固定合成 benchmark 的核心指标不超过阈值，包括 Chamfer、Hausdorff 和 2px 容差 F1，并验证真值图例排除和图像启发式图例排除都能显著降低污染误差。

生成黑色/灰色曲线共存的基准图：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name achromatic_curves --palette achromatic
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\achromatic_curves --mode predicted-mask --mask_out temp\achromatic_predicted_masks --out temp\achromatic_pred_benchmark.json
```

生成局部遮挡/断裂曲线基准图：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name local_occlusion_curves --palette basic --local_occlusion
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\local_occlusion_curves --mode predicted-mask --mask_out temp\local_occlusion_predicted_masks --out temp\local_occlusion_pred_benchmark.json
```

生成轻度交叉曲线基准图：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name crossing_curves --palette basic --crossing_curves
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\crossing_curves --mode predicted-mask --mask_out temp\crossing_predicted_masks --out temp\crossing_pred_benchmark.json
```

生成图例位于绘图区内的基准图，并对比是否排除图例区域：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name legend_inside_curves --palette basic --legend_inside
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\legend_inside_curves --mode predicted-mask --mask_out temp\legend_inside_masks_no_exclude --out temp\legend_inside_no_exclude.json
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\legend_inside_curves --mode predicted-mask --exclude_legend --mask_out temp\legend_inside_masks_exclude --out temp\legend_inside_exclude.json
```

当前 `lines.py` 是线条提取的第一版基线：

- 对 mask 二值化。
- 执行骨架化。
- 将骨架像素构造成 8 邻域图。
- 检测端点和交叉/分叉点。
- 对单连通曲线追踪主路径。
- 对虚线/点线等多连通分量，先逐片段追踪，再按综合连接评分排序拼接。
- 综合连接评分当前以距离、端点切线方向、下一段切线方向和轻量曲率连续性为默认约束，并保留 X 方向重叠、超长 gap 和短片段惩罚的可配置权重，减少“更近但急转弯”的错误连接。
- 对合理距离、整体角度和端点切线角度内的片段间隔执行直线 gap linking；端点切线角度默认阈值为 55 度，可通过 `--max_gap_tangent_angle` 调整。
- 可通过 `--tangent_gap_interpolation` 试用切线引导 Hermite 补全；当前 benchmark 显示它对现有合成 suite 不如默认直线补全稳定，因此不是默认策略。
- 在路径追踪前执行短毛刺剪枝，删除端点到交叉点之间的短分支，降低 marker、抗锯齿残留、网格/文字污染造成的伪分叉影响；可通过 `--max_spur_length` 调整，或用 `--no_spur_pruning` 关闭。
- 在 `completed_ranges` 和 `completed_pixel_count` 中记录补全位置和数量。
- 在 `confidence_per_point` 中记录点级置信度；原始观测点为高置信度，补全点为低置信度。

这能为后续的方向约束、曲率约束、虚线间距建模和机器学习 centerline 输出提供统一路径结构。

当前 `quality.py` 已支持多层基础指标：

- 坐标轴检测误差：绘图区边界误差、原点误差、轴终点误差。
- 曲线路径误差：Chamfer distance、Hausdorff distance、pred-to-truth 和 truth-to-pred 距离分布。
- observed/completed 分离误差：分别评估真实观测点和补全点相对真值曲线的偏差。
- `benchmark.py` 在 predicted-mask 模式下额外输出 mask 级指标：IoU、Precision、Recall、F1，以及 2px 容差版 Precision/Recall/F1。
- 数据空间误差：`quality.py --data_series` 和固定 benchmark suite 会输出 `data_y_mae`、`data_y_rmse`、`data_y_max_abs_error`、`data_y_p95_abs_error`、`data_r2_at_pred_x` 和 `data_x_coverage_ratio`。

其中 truth-to-pred 距离对虚线和遮挡尤其重要，因为它能反映真值曲线中有多少区域没有被提取路径覆盖。

线条 mask 属于细线目标，原始 IoU 会受到线宽和抗锯齿差异明显影响。因此后续判断“曲线像素是否提到正确位置”时，应同时看 2px 容差版 F1；原始 IoU 更适合观察 mask 是否过宽、污染是否过多。

当前纯图形学分辨基线在合成图上的参考结果：

```text
basic_curves predicted-mask:
  mean_chamfer_distance_px ≈ 0.85
  mean_truth_to_pred_px    ≈ 0.89
  mean_mask_tolerant_f1    ≈ 0.89
  mean_data_y_rmse         ≈ 0.012
  mean_data_r2_at_pred_x   ≈ 0.9997

achromatic_curves predicted-mask:
  mean_chamfer_distance_px ≈ 0.87
  mean_truth_to_pred_px    ≈ 0.92
  mean_mask_tolerant_f1    ≈ 0.89
  mean_data_y_rmse         ≈ 0.012
  mean_data_r2_at_pred_x   ≈ 0.9997

local_occlusion_curves predicted-mask:
  mean_chamfer_distance_px ≈ 0.96
  mean_hausdorff_distance_px ≈ 13.8
  mean_mask_tolerant_f1    ≈ 0.89
  mean_completed_point_ratio ≈ 0.32
  mean_data_y_rmse         ≈ 0.012
  mean_data_r2_at_pred_x   ≈ 0.9997

crossing_curves predicted-mask:
  mean_chamfer_distance_px ≈ 0.94
  mean_hausdorff_distance_px ≈ 8.8
  mean_mask_tolerant_f1    ≈ 0.88
  mean_data_y_rmse         ≈ 0.0095
  mean_data_r2_at_pred_x   ≈ 0.9994

legend_inside_curves predicted-mask，不排除图例:
  mean_chamfer_distance_px ≈ 5.13
  mean_hausdorff_distance_px ≈ 252.40
  mean_mask_tolerant_f1    ≈ 0.87
  mean_data_y_rmse         ≈ 0.55
  mean_data_r2_at_pred_x   ≈ 0.15

legend_inside_curves predicted-mask，排除合成图例区域:
  mean_chamfer_distance_px ≈ 0.85
  mean_hausdorff_distance_px ≈ 6.40
  mean_mask_tolerant_f1    ≈ 0.89
  mean_data_y_rmse         ≈ 0.012
  mean_data_r2_at_pred_x   ≈ 0.9997
```

`achromatic_curves` 用于验证黑色曲线、灰色曲线和彩色曲线共存时的分辨能力。当前策略包括：

- 在绘图区内缩后提取颜色，降低坐标轴边框干扰。
- 对黑色/灰色等低色度曲线单独抽取 achromatic prototype。
- 对灰色曲线 mask 扣除黑色像素的膨胀邻域，避免吃进黑线抗锯齿边缘。
- 删除长而薄的网格/边框组件。

图例污染处理已经开始：

- `legend.py` 根据绘图区内 OCR 文本簇给出保守的 legend bbox 候选。
- `legend.py` 也支持图像启发式检测常见带框内置图例，可识别 upper-left、lower-right 等角落位置。
- `legend.py` 也支持右上角无框图例簇检测，用于排除真实灰度样例中由图例文字和样本线造成的 mask 污染。
- `pipeline.py` 在开启 OCR 和颜色提取时，会将检测到的 legend bbox 传给颜色提取模块作为排除区域。
- `colors.py` 和 `masks.py` 均支持 `exclude_regions`，用于在颜色原型提取或 mask 生成时排除图例区域。
- `synthetic.py` 支持 `--legend_inside`，用于稳定复现绘图区内图例污染。
- `synthetic.py` 支持 `--local_occlusion`，用于稳定复现曲线短遮挡/断裂并验证补全质量。
- `synthetic.py` 支持 `--crossing_curves`，用于稳定复现颜色可分的轻度交叉曲线。
- `benchmark.py` 的 predicted-mask 模式支持 `--exclude_legend`；固定 suite 会对 `legend_inside_curves` 同时输出排除前和排除后的指标。

该能力目前分为两层：

```text
真实 pipeline：使用 OCR 文本簇启发式检测 legend bbox。
真实 pipeline：使用图像启发式检测内置带框 legend bbox。
真实 pipeline：使用右上角无框图例簇启发式检测 legend bbox。
合成 benchmark：使用已知合成图例区域作为排除框，保证图例污染前后对比可复现。
```

现阶段该模块的目标不是完整解析图例，而是先证明“图例区域应在颜色原型提取和 mask 生成前被排除”。完整图例解析，包括图例样本、曲线标签绑定、同色不同线型识别，仍属于后续专题。

## 已成型流程：CSV 校正

CSV 校正部分当前主要服务于 WebPlotDigitizer / Automeris 这类外部工具导出的点数据。它不直接负责识别图像，而是将已经提取出的点转换为可用数据。

当前流程：

```text
CSV 输入
-> 读取前两列点坐标
-> 可选四角仿射校正
-> 越界点处理
-> 归一化坐标映射到物理坐标
-> 按 X 排序
-> 可选异常点清洗
-> 可选插值或拟合
-> Plotly 预览
-> corrected CSV 导出
```

现有能力：

- 线性、log10、倒数坐标轴映射。
- 允许外推、裁剪至边界、重新归一化。
- MAD 和 Savitzky-Golay 离群点清洗。
- 线性插值、PCHIP、B 样条和高斯过程回归。
- NiceGUI 页面上传、预览和下载。

后续 CSV 模块的定位不是被本地提取替代，而是成为步骤 6 的后处理内核：

```text
本地提取出的像素点 / 归一化点
-> 数据空间映射
-> 清洗与校正
-> 质量报告
-> 标准化导出
```

## 目标完整工作流

最终项目目标是从图像直接恢复数据。建议的完整流程如下：

```text
0. 图像预处理
1. 全图 OCR
2. 坐标轴与绘图区检测
3. 九宫格区域归类
4. 坐标刻度解析
5. 图例与曲线特征提取
6. 曲线像素提取与实例分割
7. 曲线补全与中心线追踪
8. 曲线像素点数值化
9. 数据空间映射、校正与导出
```

每一步都应输出可验证的中间结果和置信度，避免整套系统变成不可调试的黑盒。

## 1. 文字内容识别

目标：

- 对当前图像执行 OCR。
- 保存所有文字框、文本内容、置信度和位置。
- 后续根据绘图区和九宫格位置，将文字分类为刻度、轴标题、图例、标题、注释或噪声。

当前原型：

- `tests/test_ocr.py`
- `tests/test_worker.py` 中也实现了按区域 OCR。

当前方法：

- 使用 RapidOCR。
- OCR 前加白边，避免边缘文字被截断。
- 放大图像，提升小字号识别率。
- OCR 后将检测框还原到原图坐标系。
- 对粘连数字做简单拆分。
- 对几何符号和部分垂直文本做过滤。

后续重点：

- 不要过早丢弃垂直文本，Y 轴标题可能是旋转文本。
- 建立文字语义分类：

```text
tick_x, tick_y, axis_label_x, axis_label_y, title, legend_label, annotation, noise
```

- 支持负号、科学计数法、log 标记、单位和上下标。
- 对关键区域进行二次 OCR，而不是只依赖全图 OCR。

## 2. 坐标轴提取与工作区划分

目标：

- 检测或推断绘图区 `plot_area`。
- 识别主 X 轴、主 Y 轴、原点、绘图区四边界。
- 将整张图划分为九宫格，用于 OCR 和图例归类。

当前原型：

- `tests/test_pre.py`
- `tests/test_worker.py`

当前方法：

- 灰度化或最小通道提取。
- 全局阈值、Otsu 或自适应阈值二值化。
- 可选形态学闭运算连接断线。
- 使用 `cv2.HoughLinesP` 检测线段。
- 按角度区分水平线和垂直线。
- 合并共线且端点接近的短线。
- 选最靠下的长水平线作为 X 轴。
- 选最靠左的长垂直线作为 Y 轴。
- 用局部像素密度精修轴线位置。

九宫格设计：

```text
左上角区域 | 上方标题/顶部图例 | 右上角区域
Y刻度/标签 | 绘图区          | 右侧图例/右轴
左下角区域 | X刻度/标签      | 右下角区域
```

后续重点：

- 多数图不一定有四条显式坐标轴，应优先提取或推断绘图区四边界。
- 不能只依赖长直线，还要结合 OCR 数字分布、刻度线、网格线和曲线像素分布。
- 支持无边框图、双 Y 轴、右轴、上轴、反向轴、log 轴和轻微旋转图。

## 3. 线条特征提取

目标：

- 获取每条曲线的身份特征。
- 有图例时从图例解析颜色、线型、marker 和标签。
- 无图例时直接从绘图区聚类曲线特征。

当前原型：

- `tests/test_colors.py`
- `tests/test_worker.py` 中对区域 0 进行颜色提取。

当前方法：

- 将图像转为 Lab 色彩空间。
- 通过亮度和色度过滤背景、文字、坐标轴、灰色网格。
- 对剩余区域用 `cv2.floodFill` 提取连通域。
- 计算连通域平均 Lab。
- 按面积排序后做颜色合并。
- 使用宽松 L 容差和严格 ab 容差，处理抗锯齿和同色深浅变化。
- 根据面积占比过滤噪声。

建议输出：

```text
curve_id
label
rgb/lab
line_style: solid/dashed/dotted/dashdot/unknown
marker_style
confidence
source: legend/direct
```

后续重点：

- 黑色曲线不能简单按低亮度过滤，否则会被当作文字/轴线丢弃。
- 灰色曲线和网格线需要通过位置、线宽、连通性和图例辅助区分。
- 同色不同线型、同色不同 marker 需要补线型和 marker 识别。
- 图例线段短、面积小，不能只按全图面积阈值判断。

## 4. 完整线条捕捉与补全

目标：

- 对每条曲线得到完整像素轨迹。
- 处理断裂、遮挡、虚线、网格覆盖和曲线交叉。
- 对不确定区域给出置信度。

当前原型：

- `tests/test_lines.py`

当前方法：

- 根据目标颜色 Lab 生成二值 mask。
- 形态学开运算去噪，闭运算弥合小断裂。
- 使用 `skimage.morphology.skeletonize` 做骨架化。
- 使用连通域提取曲线片段。
- 取片段端点，构造距离矩阵。
- 优先使用 scipy 匈牙利算法匹配端点，没有 scipy 时降级为贪心匹配。
- 将距离小于阈值的端点直接连接。

当前问题：

- 连通域点集没有真正按曲线拓扑排序。
- 当前片段起点和终点可能不是真实曲线端点。
- 端点补全主要看距离，缺少方向、曲率和上下文约束。
- 高度重叠时，图形学方法容易失效。
- 曲线交叉处容易发生错误归属。

机器学习评价：

有必要考虑引入机器学习，但不建议一开始做端到端图片转 CSV。更稳妥的路线是：

```text
图形学负责高置信度区域
机器学习负责困难区域的分割、归属和补全
规则系统负责坐标映射和质量校验
```

机器学习适合介入的位置：

- 曲线像素语义分割。
- 曲线实例分割。
- 交叉点归属判断。
- 遮挡区域走向预测。
- 病态重叠区域的置信度估计。

训练数据建议：

- 扩展 `tests/test_draw.py`。
- 自动生成图像、每条曲线真值 mask、真实曲线坐标和遮挡标签。
- 用合成数据先建立 benchmark，再考虑真实数据微调。

重要限制：

高度重叠的病态问题可能本身信息不足。若两条曲线颜色、线型和位置长时间重合，任何方法都只能推断，不能保证唯一正确。因此系统必须输出置信度，并允许人工修正。

## 5. 子线条数值化

目标：

- 将每条曲线的像素轨迹转为有序像素点序列。
- 按合适的采样密度生成离散点。

这一步不是简单保存 mask 像素，而是要解决“有序化”：

```text
curve mask / skeleton
-> centerline
-> ordered pixel polyline
-> sampled pixel points
```

后续重点：

- 实现真实 8 邻域图遍历。
- 检测端点、分叉点、交叉点。
- 对局部切线和曲率连续性建模。
- 对重复 X 做合并或标记。
- 判断曲线是否满足 `y = f(x)`。
- 对回折曲线、闭合曲线、参数曲线保留路径顺序，而不是强制按 X 排序。
- 控制重采样密度，避免过密点影响后续拟合。

## 6. 数值映射还原

目标：

- 将曲线像素点映射回原始数据空间。
- 默认输出校正后的结果。
- 将原 CSV 校正逻辑直接集成到此步骤。

映射所需信息：

```text
x_pixel_min, x_pixel_max
y_pixel_min, y_pixel_max
x_data_min, x_data_max
y_data_min, y_data_max
x_scale: linear/log/inverse
y_scale: linear/log/inverse
```

数据范围来源：

- OCR 识别到的刻度文本。
- 坐标轴像素位置。
- 用户手动修正。
- 图像生成 benchmark 的真值。

建议流程：

```text
像素点
-> 像素坐标归一化
-> 数据空间映射
-> 越界检查
-> 异常点清洗
-> 可选重采样和拟合
-> 质量报告
-> corrected CSV / Excel 导出
```

这一步应复用并重构 `src/csv_processor.py`，使其同时支持：

- WebPlotDigitizer 已导出的 CSV。
- 本地工具输出的归一化点。
- 本地工具输出的像素点。

## 优先级规划

短期目标：

1. 明确每一步的标准输入输出结构。
2. 将 `tests/test_worker.py` 改造成可输出 JSON 的 pipeline。
3. 把坐标轴检测结果封装为统一对象：

```text
plot_area, origin, x_axis, y_axis, confidence
```

4. 将 OCR 结果按九宫格归类。
5. 扩展 `test_draw.py`，生成图像时同步输出真实曲线数据。

中期目标：

1. 完成刻度 OCR 语义解析。
2. 实现图例解析和曲线身份特征提取。
3. 改造线条提取，实现真实路径追踪。
4. 完成曲线像素点到有序数据点的转换。
5. 将 CSV 校正模块集成到图像提取 pipeline 的最终步骤。

长期目标：

1. 引入机器学习处理曲线实例分割和病态补全。
2. 建立合成数据训练集和 benchmark。
3. 支持人工修正入口。
4. 输出完整质量报告，包含每条曲线的置信度、补全区域和误差估计。

## 质量控制原则

本项目最终输出的是科学数据，因此不能只追求视觉上看起来正确。每个阶段都应保存：

- 原始输入。
- 中间结果。
- 参数。
- 置信度。
- 可视化 debug artifact。
- 错误和警告。

最终输出除数据表外，还应包含质量报告：

```text
坐标轴检测置信度
OCR 刻度解析置信度
曲线颜色/线型识别置信度
曲线补全比例
越界点比例
清洗删除点数
拟合残差指标
人工修正记录
```

这将使 Graph2Data 能够逐步从外部网站导出依赖，过渡到可信的本地图表数据恢复工具。

## 工程推进路线

本项目不建议直接从复杂算法或机器学习开始。更稳妥的推进方式是：

```text
工程化现有原型
-> 建立可复现 benchmark
-> 固化坐标轴、OCR、映射基础能力
-> 改造图形学线条提取
-> 引入机器学习处理困难区域
-> 最后完善 GUI、人工修正和批处理
```

核心原因是：如果没有统一数据结构、真值数据和误差指标，每个模块都可能看起来有效，但无法判断整体结果是否真正变好。

## 工程实现阶段规划

本项目的实现过程按“先闭环、再提高鲁棒性、最后产品化”的顺序推进。每个阶段都应留下可运行命令、结构化输出、debug artifact 和自动化质量门，避免算法改动只能靠肉眼判断。

### 阶段 A：工程基线与最小闭环

阶段目标：

建立正式包结构和统一数据接口，使项目从松散原型脚本进入可维护的工程形态。

工作内容：

- 将 `tests/` 下成熟的图像处理原型迁移到 `src/graph2data/`。
- 建立 `models.py`，统一 `AxisDetection`、`PlotArea`、`CurvePrototype`、`CurveMask`、`CurvePath`、`DataSeries`、`PipelineResult` 等核心对象。
- 建立 `pipeline.py`，串联图像读取、坐标轴检测、版面划分、颜色原型提取、mask 提取、path 提取、数据映射和 artifact 输出。
- 建立 `image_io.py`，统一 JSON、目录和图像 artifact 写入方式。
- 保留旧的 CSV 校正工具，但将其定位为后续数据后处理内核。

期望结果：

- 能通过一条命令完成 `图像 -> mask -> path -> CSV` 的最小闭环。
- pipeline 输出结构化 JSON，而不是只依赖控制台打印或窗口展示。
- 所有阶段性结果都有明确字段和 artifact 路径。

验收条件：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1_mapped.json --map_data --x_min 0 --x_max 100 --y_min -10 --y_max 10 --artifact_dir temp\pipeline_test1_mapped_artifacts --debug_artifacts
```

应生成：

```text
temp/pipeline_test1_mapped.json
temp/pipeline_test1_mapped_artifacts/masks/*.png
temp/pipeline_test1_mapped_artifacts/debug/overview.png
temp/pipeline_test1_mapped_artifacts/debug/paths_overlay.png
temp/pipeline_test1_mapped_artifacts/data/curves.csv
temp/pipeline_test1_mapped_artifacts/quality/report.json
```

当前状态：基本完成，后续只做增量补强。

### 阶段 B：Benchmark 与质量门

阶段目标：

建立可复现、可量化、可失败的回归测试体系，使算法改动可以被客观比较。

工作内容：

- 完善 `synthetic.py`，生成图像、坐标轴真值、曲线真值数据、曲线真值 mask。
- 完善 `quality.py`，支持坐标轴误差、mask 指标、path 指标和数据空间指标。
- 完善 `benchmark.py`，支持固定 suite、单 case 评估、图例排除对比、检测图例排除对比。
- 建立 pytest 回归测试，覆盖 pipeline 闭环、mapping 正确性、图例检测、gap linking 和 benchmark 阈值。
- 将 `pixi run python -m pytest -q` 作为每次算法改造后的基本质量门。

期望结果：

- 任意一次算法改动都能回答“变好了还是变坏了”。
- 图例污染、黑灰曲线、多色曲线、虚线补全等核心风险都有基准场景。
- pytest 不再出现 `no tests ran`，而是运行真实测试。

验收条件：

```powershell
$env:PYTHONPATH='src'
pixi run python -m pytest -q
pixi run python -m graph2data.benchmark --suite --suite_out temp\suite_check
```

期望：

```text
pytest 全部通过
basic_curves / achromatic_curves / local_occlusion_curves / crossing_curves / legend_inside_curves 均有稳定指标
legend_inside_curves 中 image_heuristic 图例排除能显著降低 Hausdorff
```

当前状态：已建立，后续随着新场景继续扩充。

### 阶段 C：可视化 Debug 与人工诊断能力

阶段目标：

让每个算法阶段的错误都能被定位，而不是只看到最终 CSV 错误。

工作内容：

- 输出坐标轴、plot area、legend bbox、颜色 prototype、mask、path 和数据映射结果。
- 增加 `--debug_artifacts`，生成 overview、全局 path overlay、逐曲线 path overlay。
- 在 `quality/report.json` 中输出轴、图例、mask/path/data 数量、补全比例、低置信度点比例和逐曲线摘要。
- 后续补充 skeleton overlay、候选坐标轴 overlay、OCR box overlay、图例样本 overlay。
- 在 `PipelineResult.artifacts` 中记录所有 artifact 路径，便于 GUI 或批处理报告引用。

期望结果：

- mask 污染、路径断裂、图例误检、坐标轴偏移等问题能通过图片直接定位。
- benchmark 输出不只是 JSON 指标，也能保留关键 debug 图。

验收条件：

```text
debug/overview.png          能看到 plot area、坐标轴和图例框
debug/paths_overlay.png     能看到所有曲线的 path 覆盖情况
debug/paths/<curve_id>.png  能逐曲线检查路径质量
masks/<curve_id>.png        能单独检查颜色分割结果
quality/report.json         能看到单张图的基础质量摘要
```

当前状态：基础 artifact 已完成，skeleton、OCR、候选轴线 overlay 待补。

### 阶段 D：图例检测与图例解析

阶段目标：

降低图例对颜色原型、mask 和路径追踪的污染，并逐步建立曲线标签绑定能力。

工作内容：

- 当前阶段先完成图像启发式图例检测：识别常见内置带框图例，支持 upper-left、lower-right 等角落位置。
- 将检测到的图例 bbox 作为 `exclude_regions`，在颜色原型和 mask 提取前排除。
- 在 benchmark 中同时评估三种模式：不排除图例、使用真值图例排除、使用图像启发式图例排除。
- 下一步增加图例样本提取：从图例区域识别线段颜色、线型和标签文字。
- 再下一步建立标签绑定：将图例样本与曲线 prototype/path 绑定，输出曲线名称。

期望结果：

- 绘图区内图例不再显著拉高 path Hausdorff。
- 检测图例排除的指标接近真值图例排除。
- 普通曲线密集区域不应被误检为图例。

验收条件：

```text
legend_inside_curves，不排除图例:
  Hausdorff 显著偏高

legend_inside_curves，image_heuristic 图例排除:
  detected_legend_count >= 1
  mean_hausdorff_distance_px 回到正常范围
```

当前状态：第一版图像启发式排除已完成；完整图例解析和标签绑定待做。

### 阶段 E：曲线颜色、Mask 与路径追踪算法

阶段目标：

提高从曲线像素到有序路径的稳定性，降低断线、误连、交叉和重叠带来的错误。

工作内容：

- 优化 `colors.py`：提高黑色、灰色、低饱和曲线和抗锯齿边缘的 prototype 稳定性。
- 优化 `masks.py`：降低网格线、文字、坐标轴和图例残留污染。
- 当前 `masks.py` 已加入保守的边缘极小刻度残片过滤；该规则只处理紧贴边缘的微小组件，避免误删虚线/点线曲线端点。
- 优化 `lines.py`：将多连通分量按距离、方向、端点切线和曲率连续性综合排序，并预留 X 重叠、超长 gap 和短片段惩罚权重。
- 当前已增加 gap linking 的端点切线角约束，可通过 `--max_gap_tangent_angle` 调整。
- 当前已增加骨架短毛刺剪枝，减少伪分叉对主路径追踪的影响。
- 当前已增加实验性 marker-like 紧凑组件过滤，可通过 `lines.py --filter_marker_like` 或 pipeline 的 `--line_filter_marker_like` 启用；默认关闭，防止误删虚线/点线的真实短片段。
- 当前已加入局部遮挡/断裂 synthetic case，用于验证 gap linking 对短缺失段的补全质量。
- 当前已加入轻度交叉 synthetic case，用于验证颜色可分交叉区域的 mask、path 和数据空间稳定性。
- 后续加入 marker 曲线、同色不同线型、虚线密度变化等 synthetic case。

期望结果：

- 虚线/点线能保持较高 truth-to-pred 覆盖率。
- 不合理跨段连接减少。
- 曲线交叉区域能保留明确 warnings 或低置信度，而不是静默输出错误路径。

验收条件：

```text
basic_curves:
  mean_chamfer_distance_px < 1.25
  mean_mask_tolerant_f1 > 0.85

achromatic_curves:
  valid_curve_count == curve_count

新增困难场景:
  指标有明确阈值
  出现歧义时有 warnings 或低置信度记录
```

当前状态：路径追踪、短毛刺剪枝、综合片段连接评分、gap linking、局部短遮挡 benchmark、轻度交叉 benchmark、保守 mask 残片过滤和实验性 marker-like 过滤已可用；复杂交叉、同色线型归属、marker/线型联合分离、线型周期建模和长遮挡补全待做。

### 阶段 F：坐标轴、OCR 与数据空间映射

阶段目标：

让系统从“像素路径”真正进入“数据坐标”，并逐步减少用户手动输入坐标范围的依赖。

工作内容：

- 当前 `mapping.py` 已支持根据 plot area 和用户提供的 `DataRange` 将 `CurvePath` 映射为 `DataSeries`。
- 当前 pipeline 已支持 `--map_data` 并输出 `data/curves.csv`。
- 当前 `quality.py` 和 `benchmark.py` 已支持数据空间 RMSE、MAE、Max Error、P95 Error、R2 和 X 覆盖率评估。
- 下一步实现 tick OCR 语义解析：识别刻度文字、单位、数值范围和 log/inverse 轴。
- 将 OCR 结果、坐标轴检测结果和 layout 区域合并为可靠的坐标系推断。
- 后续补充单调性检查、越界比例和基于 OCR 的自动坐标范围推断。

期望结果：

- 用户提供坐标范围时，能稳定输出数据坐标 CSV。
- 对合成 benchmark，能直接评估数据空间误差。
- 对真实图，能逐步从手动范围输入过渡到 OCR 辅助识别。

验收条件：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\mapped.json --map_data --x_min 0 --x_max 100 --y_min -10 --y_max 10 --artifact_dir temp\mapped_artifacts
```

应输出：

```text
data_series 非空
data/curves.csv 存在
每个 DataPoint 包含 x/y、pixel_x/pixel_y、confidence、completed
```

当前状态：手动范围映射和数据空间质量评估已完成；OCR 刻度语义解析、自动坐标范围推断、单调性和越界质量指标待做。

### 阶段 G：后处理、清洗、拟合与导出

阶段目标：

将本地图像提取结果和已有 CSV 校正能力统一起来，输出可直接分析的数据表和质量报告。

工作内容：

- 将 `csv_processor.py` 中的坐标映射、清洗、拟合和导出能力重构为可被 pipeline 调用的后处理模块。
- 支持对 `DataSeries` 做去重、排序、离群点剔除、插值、PCHIP、B-spline、GPR 等处理。
- 输出 corrected CSV、可选 Excel、质量报告 JSON。
- 在报告中记录补全比例、低置信度点比例、清洗删除点数、拟合残差和 warnings。

期望结果：

- 外部 CSV 工作流和本地图像提取工作流共用同一套后处理逻辑。
- 用户能选择“原始提取点”或“清洗拟合后曲线”导出。
- 每条曲线都有质量摘要。

验收条件：

```text
输入 DataSeries
-> 输出 raw CSV
-> 输出 corrected CSV
-> 输出 quality_report.json
-> GUI/CLI 均可调用
```

当前状态：CSV 校正工具独立可用；正式接入 pipeline 待做。

### 阶段 H：真实图鲁棒性、人工修正与产品化

阶段目标：

让工具从研发基线过渡到可用于真实论文图和实验图的半自动数据恢复工具。

工作内容：

- 建立真实图测试集，覆盖论文截图、低分辨率压缩图、浅色曲线、复杂图例、双轴图、log 坐标图。
- 为无法自动确定的区域输出 ambiguous 状态，而不是强行给出错误结果。
- 增加人工修正入口：plot area 修正、坐标范围修正、图例排除框修正、曲线路径局部修正。
- 将 CLI、批处理和 GUI 整合，支持批量图像处理和报告导出。
- 对困难区域逐步引入机器学习分割或交互式修正模型，但保留可解释中间结果。

期望结果：

- 真实图处理失败时能说明失败原因和低置信度区域。
- 用户可以修正关键中间结果并重新运行后续阶段。
- 输出不只是 CSV，还包含质量报告和 debug artifact。

验收条件：

```text
真实图批处理可运行
失败样例有明确 warnings
人工修正后可复跑后续步骤
导出数据表 + 质量报告 + debug artifact
```

当前状态：尚未进入产品化阶段。

### 阶段 I：机器学习增强

阶段目标：

在图形学基线和 benchmark 足够稳定后，用机器学习处理传统规则难以覆盖的分割、归属和补全问题。

工作内容：

- 基于 synthetic 和真实标注数据构建训练集。
- 训练或接入曲线实例分割模型，输出曲线 mask 或 centerline probability map。
- 使用模型处理交叉、重叠、遮挡、同色不同线型等困难区域。
- 将模型输出接入现有 `CurveMask`、`CurvePath`、`DataSeries` 结构，而不是绕过工程闭环。
- 继续用 benchmark 和真实图集评估模型收益，避免模型在简单场景上退化。

期望结果：

- 机器学习只替换或增强困难模块，不破坏现有可解释 pipeline。
- 模型输出有置信度和 fallback 策略。
- 简单图仍可由图形学稳定处理，复杂图由模型辅助。

验收条件：

```text
模型输出可转换为 CurveMask / CurvePath
简单 benchmark 不退化
困难 benchmark 指标显著改善
低置信度区域能回传给人工修正流程
```

当前状态：暂不作为主路径，待前述阶段稳定后进入。

### 阶段 0：工程化原型

将 `tests/` 下的实验脚本逐步迁移为正式模块。建议结构：

```text
src/graph2data/
  __init__.py
  pipeline.py
  models.py
  image_io.py
  preprocess.py
  ocr.py
  axes.py
  layout.py
  legend.py
  colors.py
  lines.py
  mapping.py
  quality.py
  export.py
```

优先定义统一数据结构：

```text
OCRTextBox
AxisDetection
PlotArea
LayoutRegions
CurvePrototype
CurveMask
CurvePath
DataSeries
QualityReport
PipelineResult
```

每个模块应返回结构化对象，不应直接依赖 `print`、`imshow`、`plt.show`。调试结果应作为 artifact 保存。

### 阶段 1：建立 Benchmark

扩展 `tests/test_draw.py`，使其不仅生成测试图，还同步生成真值：

```text
image.png
truth_axes.json
truth_curves.json
truth_masks/
truth_data.csv
```

覆盖场景：

```text
单色/多色曲线
黑色曲线
灰色曲线
实线/虚线/点线
marker 曲线
有/无图例
有/无网格
曲线交叉
局部遮挡
高度重叠
log 坐标轴
双 Y 轴
低分辨率和压缩图
```

评价指标：

```text
坐标轴：像素误差、绘图区 IoU
OCR：刻度识别准确率、数值解析准确率
颜色/图例：curve prototype 匹配率
mask：IoU、Precision、Recall、F1、2px tolerant Precision/Recall/F1
path：Chamfer distance、Hausdorff distance
数据：RMSE、MAE、Max Error、R2
```

没有 benchmark 之前，不建议引入机器学习作为主路径。

### 阶段 2：坐标轴与版面理解

坐标轴检测建议采用规则和图形学组合，而不是一开始机器学习。候选方法：

```text
Canny / threshold
+ HoughLinesP
+ connected components
+ OCR tick distribution
+ layout scoring
```

从“固定取最下水平线和最左垂直线”升级为候选打分：

```text
score(axis_pair) =
  长度分
+ 正交分
+ 位置分
+ 刻度文字贴近分
+ 网格线一致性分
+ 曲线像素落入区域分
```

输出应包含最佳结果、候选列表、置信度和失败原因。

### 阶段 3：OCR 与刻度解析

OCR 可继续采用 RapidOCR，重点放在后处理：

```text
全图 OCR
-> 根据 plot_area 分区
-> 文本类型分类
-> 数值文本解析
-> tick 位置拟合
-> 坐标尺度判断
```

数值解析应支持：

```text
-1.2
1e-3
10^-2
×10^3
1,000
log / ln
```

线性轴使用线性回归拟合 `pixel -> data`；log 轴先对数据值取 `log10` 再拟合。

### 阶段 4：图例与曲线特征

曲线身份识别分两种路径：

```text
有图例：legend-first
无图例：plot-first
```

有图例时，寻找短线段、marker 和相邻文本组合，生成曲线原型。无图例时，直接在绘图区聚类曲线颜色和线型。

颜色提取建议从当前 floodFill 手写聚类逐步升级为：

```text
有效像素采样
-> Lab 特征
-> DBSCAN / HDBSCAN / 层次聚类
-> 聚类中心
-> 反投影生成每类 mask
```

输出标准曲线原型：

```text
curve_id
label
rgb/lab
line_style
marker_style
confidence
source: legend/direct
```

### 阶段 5：线条提取算法选型

线条提取是本项目核心难点。建议采用分层方案。

#### 基线方案：图形学 + 图搜索

适用于颜色可分、遮挡较轻的图表：

```text
curve prototype
-> 颜色/线型 mask
-> 去除坐标轴、文字、网格
-> skeletonize
-> graph construction
-> endpoint / junction detection
-> path tracing
-> gap linking
-> centerline smoothing
```

关键技术：

- 彩色曲线：Lab 色差阈值。
- 黑色曲线：绘图区内黑色像素减去坐标轴、文字、刻度和图例。
- 灰色曲线：结合方向、线宽、连通性排除网格。
- 骨架化：`skimage.morphology.skeletonize` 或 `cv2.ximgproc.thinning`。
- 图建模：骨架像素为节点，8 邻域为边。
- 关键点：`degree == 1` 为端点，`degree >= 3` 为分叉或交叉。

断点连接不应只按距离，应使用代价函数：

```text
cost =
  distance_weight * endpoint_distance
+ angle_weight * tangent_angle_diff
+ curvature_weight * curvature_change
+ color_weight * color_consistency
+ crossing_penalty
```

#### 虚线和点线

虚线应按 stroke segment 处理：

```text
同色短片段检测
-> 局部方向估计
-> 片段级图构建
-> 按方向、间距、颜色匹配连接
```

segment 结构：

```text
points
center
length
tangent
color
bbox
```

#### 交叉和重叠

交叉点需要特殊处理：

- 检测 junction。
- 提取进入和离开的分支。
- 用切线方向做连续性配对。
- 颜色不同则按颜色直接分开。
- 颜色相同但线型不同则按线型节奏或 marker 区分。
- 信息不足时标记为 ambiguous，不强行输出伪确定结果。

#### 机器学习方案

机器学习有必要引入，但不建议端到端图片转 CSV。更合适的路线：

```text
图形学负责高置信度区域
机器学习负责困难区域的分割、归属和补全
规则系统负责坐标映射和质量校验
```

推荐选型：

1. 语义分割：U-Net、DeepLabV3+、SegFormer-B0/B1。
2. 中心线预测：输出 centerline probability heatmap。
3. 条件分割：输入图像 + 曲线特征或图例 patch，输出指定曲线 mask。

推荐路线：

```text
第一阶段：合成数据训练 U-Net / SegFormer 做曲线语义 mask
第二阶段：扩展为多通道实例 mask
第三阶段：做 prototype-conditioned curve segmentation
第四阶段：对遮挡和重叠区域预测 centerline heatmap
```

不建议训练“图片直接输出 CSV”的端到端回归模型。科学数据恢复需要可解释中间结果。

### 阶段 6：数值化与映射

线条提取模块最终应输出：

```text
CurvePath {
  curve_id
  pixel_points_ordered
  observed_mask
  completed_mask
  ambiguous_regions
  confidence_per_point
}
```

数值化流程：

```text
ordered pixel path
-> 裁剪到 plot_area
-> 按 X 或路径重采样
-> 像素坐标归一化
-> 映射数据坐标
-> 清洗、拟合、导出
```

常规 `y = f(x)` 曲线可按 X 采样；参数曲线、回折曲线和闭合曲线应保留路径顺序。

## 项目难度与完成度预期

### 整体实现难度

Graph2Data 的整体难度较高。它不是单一 OCR、图像分割或曲线拟合问题，而是多个不稳定视觉任务和科学数据约束的组合：

```text
图表结构理解
OCR 语义解析
曲线实例分割
遮挡与重叠推断
像素到数据空间映射
质量评估与可追溯导出
```

其中坐标轴检测、颜色曲线提取、CSV 校正属于中等难度；OCR 刻度语义解析、黑色/灰色曲线分离、交叉点归属属于较高难度；高度重叠和严重遮挡曲线补全属于高难度甚至部分病态问题。

### 完成度预期

合理的完成度目标应分层设定：

1. 基础完成度：处理清晰、常规、彩色、多曲线但不严重重叠的科学图表。
2. 中级完成度：支持图例解析、log 轴、虚线、marker、局部遮挡和轻度交叉。
3. 高级完成度：支持黑白图、灰色曲线、复杂图例、多坐标轴和部分高度重叠。
4. 研究级完成度：对严重遮挡和高度重叠给出概率化恢复、置信度和人工修正入口。

短中期不应追求对所有论文图“一键完美提取”。更现实的目标是：

```text
常规图自动化
复杂图半自动化
病态图可诊断、可修正、可追溯
```

### 精度预期

精度取决于图像质量、曲线复杂度、坐标轴识别质量和曲线是否可分。

对清晰彩色图：

- 坐标轴定位可达到 1-3 像素级误差。
- 曲线 mask F1 有机会达到 0.90 以上。
- 最终数据 RMSE 可接近 WebPlotDigitizer 人工提取结果。

对有轻度交叉和局部遮挡的图：

- 如果颜色或线型可区分，主要曲线段可达到较高可用性。
- 遮挡段应输出补全标记和较低置信度。
- 最终数据精度可能在局部下降，但整体趋势仍可恢复。

对黑白、灰度、同色多线型图：

- 单靠颜色分割不可行，需要线型、marker、拓扑和图例辅助。
- 精度将明显依赖图像分辨率和线型可辨识度。

对高度重叠病态图：

- 不应承诺确定性高精度。
- 如果信息本身不足，算法只能基于上下文推断。
- 应输出多个候选路径、置信度和人工确认入口。

因此最终系统的理想输出不是单一 CSV，而是：

```text
数据表
+ 曲线级置信度
+ 点级置信度
+ 补全区域标记
+ ambiguous 区域
+ 质量报告
```

## 项目价值评价

### 工程价值

Graph2Data 有明确工程价值。现有 WebPlotDigitizer 这类工具强依赖人工交互，批量化和可追溯能力有限。本项目如果完成稳定 pipeline，可以提供：

- 本地图表数据恢复。
- 批量处理能力。
- 自动质量报告。
- 可保存的中间结果。
- 与 Python 科学计算生态直接集成。
- 对 WebPlotDigitizer 导出结果的后处理和校验。

即使最终不能完全自动处理所有复杂图，作为“自动提取 + 人工修正 + 质量评估”的工具也有实际价值。

### 科研价值

科研上，本项目位于文献图表理解、科学数据恢复和细粒度曲线实例分割的交叉点。它的价值不只是做一个工具，还包括：

- 建立科学图表曲线提取 benchmark。
- 研究细线条实例分割与中心线恢复。
- 研究图例和绘图区之间的视觉对应关系。
- 研究 OCR 刻度语义与坐标映射。
- 研究遮挡曲线的概率化补全和不确定性表达。

如果能系统化生成合成数据、真值 mask、真值曲线和误差评价，就具备较强的实验研究价值。

### 创新点

潜在创新点包括：

- 将图例 prototype 条件化用于曲线实例分割。
- 将图形学骨架追踪和机器学习 centerline heatmap 结合。
- 对复杂图表提取结果输出点级置信度和 ambiguous 区域，而不是只输出单一结果。
- 将 CSV 校正、曲线补全和质量报告集成为闭环。
- 用可控合成图系统性评估曲线提取算法。

### 风险评价

主要风险：

- 真实论文图风格差异极大。
- 高度重叠曲线存在不可辨识情况。
- OCR 错误会直接影响坐标映射。
- 黑白图、压缩图、扫描图会显著降低视觉特征质量。
- 机器学习需要足够多样的训练数据和清晰真值。

应对策略：

- 不追求一步到位。
- 先建立 benchmark。
- 让所有模块输出置信度和失败原因。
- 保留人工修正入口。
- 将自动化目标限定为“常规图高自动化，复杂图辅助式恢复”。

综合评价：Graph2Data 是一个实现难度高但价值明确的项目。其工程价值在于替代和增强现有人工图表数字化流程；科研价值在于细粒度图表理解、曲线实例分割和不确定性数据恢复。只要按模块化、可评估、可追溯的路线推进，该项目有希望达到较高实用完成度。

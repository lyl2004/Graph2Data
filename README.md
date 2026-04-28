# Graph2Data

Graph2Data 是一个面向科学图表的数据恢复项目，目标是从论文图、实验图或软件导出的曲线图中恢复可计算的数据表。

当前项目包含两条工作线：

1. 已成型的 CSV 校正工具：接收 Automeris / WebPlotDigitizer 等工具导出的曲线点 CSV，完成坐标映射、清洗、拟合和导出。
2. 尚在原型阶段的本地图像提取工具：尝试直接从图像中检测坐标轴、识别文字、提取曲线特征、补全曲线并还原数值。

后续工作的核心方向是逐步补完本地图像提取链路，使其最终替代网站工具导出的中间 CSV，同时保留 CSV 校正模块作为统一后处理和质量控制内核。

## 当前项目结构

```text
Graph2Data/
  assets/              静态资源、字体和样式文件
  src/
    csv_gui.py         NiceGUI 可视化入口
    csv_processor.py   CSV 坐标映射、清洗、拟合核心逻辑
    csv_create.py      合成测试图与理论曲线校验工具
    config.json        当前参数配置草稿
    tempreadme         CSV 校正模块设计记录
  tests/
    test_pre.py        坐标轴检测原型
    test_ocr.py        OCR 文本识别原型
    test_colors.py     曲线颜色提取原型
    test_lines.py      曲线线条提取与补全原型
    test_worker.py     图像分析组合工作流原型
    test_draw.py       合成测试图生成工具
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
pipeline.py     结构化 pipeline 入口
synthetic.py    合成 benchmark 生成器
quality.py      最小质量评估工具
lines.py        mask 骨架化和有序路径追踪
benchmark.py    批量 benchmark runner
```

可以用以下命令运行结构化 pipeline：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1.json --colors
```

如需开启 OCR：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.pipeline --img tests\test1.png --out temp\pipeline_test1_ocr.json --ocr --colors
```

当前新 pipeline 的目标不是一次性替代所有 `tests/` 原型，而是先建立稳定的数据接口和 JSON 输出。后续会逐步把 `tests/test_pre.py`、`test_ocr.py`、`test_colors.py`、`test_lines.py` 中成熟的算法迁移进正式模块。

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

生成黑色/灰色曲线共存的基准图：

```powershell
$env:PYTHONPATH='src'
pixi run python -m graph2data.synthetic --out benchmarks\synthetic --name achromatic_curves --palette achromatic
pixi run python -m graph2data.benchmark --case benchmarks\synthetic\achromatic_curves --mode predicted-mask --mask_out temp\achromatic_predicted_masks --out temp\achromatic_pred_benchmark.json
```

当前 `lines.py` 是线条提取的第一版基线：

- 对 mask 二值化。
- 执行骨架化。
- 将骨架像素构造成 8 邻域图。
- 检测端点和交叉/分叉点。
- 对单连通曲线追踪主路径。
- 对虚线/点线等多连通分量，先逐片段追踪，再按 X 方向拼接。
- 对合理距离和角度内的片段间隔执行第一版直线 gap linking。
- 在 `completed_ranges` 和 `completed_pixel_count` 中记录补全位置和数量。
- 在 `confidence_per_point` 中记录点级置信度；原始观测点为高置信度，补全点为低置信度。

这能为后续的方向约束、曲率约束、虚线间距建模和机器学习 centerline 输出提供统一路径结构。

当前 `quality.py` 已支持两类基础指标：

- 坐标轴检测误差：绘图区边界误差、原点误差、轴终点误差。
- 曲线路径误差：Chamfer distance、Hausdorff distance、pred-to-truth 和 truth-to-pred 距离分布。
- observed/completed 分离误差：分别评估真实观测点和补全点相对真值曲线的偏差。

其中 truth-to-pred 距离对虚线和遮挡尤其重要，因为它能反映真值曲线中有多少区域没有被提取路径覆盖。

当前纯图形学分辨基线在合成图上的参考结果：

```text
basic_curves predicted-mask:
  mean_chamfer_distance_px ≈ 0.83
  mean_truth_to_pred_px    ≈ 0.87

achromatic_curves predicted-mask:
  mean_chamfer_distance_px ≈ 0.87
  mean_truth_to_pred_px    ≈ 0.92
```

`achromatic_curves` 用于验证黑色曲线、灰色曲线和彩色曲线共存时的分辨能力。当前策略包括：

- 在绘图区内缩后提取颜色，降低坐标轴边框干扰。
- 对黑色/灰色等低色度曲线单独抽取 achromatic prototype。
- 对灰色曲线 mask 扣除黑色像素的膨胀邻域，避免吃进黑线抗锯齿边缘。
- 删除长而薄的网格/边框组件。

图例污染处理已经开始：

- `legend.py` 根据绘图区内 OCR 文本簇给出保守的 legend bbox 候选。
- `pipeline.py` 在开启 OCR 和颜色提取时，会将检测到的 legend bbox 传给颜色提取模块作为排除区域。
- `colors.py` 和 `masks.py` 均支持 `exclude_regions`，用于在颜色原型提取或 mask 生成时排除图例区域。

该能力目前是启发式的前置版本，目标是先处理明显的绘图区内图例污染。完整图例解析，包括图例样本和曲线标签绑定，仍属于后续专题。

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
mask：IoU、Precision、Recall、F1
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

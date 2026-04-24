# 铣削表面质量预测双模式框架设计

日期：2026-04-24

## 1. 背景与目标

当前项目围绕铣削表面质量预测展开，已有两套原型代码：

- `bpnn.py`：基于 `numpy` 的双网络原型，输入为切削参数和振动统计特征，输出为 `Ra` 与 8 维频域幅值。
- `milling_prediction_framework.py`：基于 `PyTorch` 的完整单文件实验原型，包含仿真数据生成、多任务网络训练和可视化。

现有原型能展示思路，但存在几个问题：

- 数据生成、特征提取、建模训练、结果输出耦合在同一脚本中，后续扩展真实数据较困难。
- 模拟数据与真实实验数据的边界不清晰，老师不容易判断哪些部分是经验设定，哪些部分是未来可替换接口。
- 缺少正式 `README.md`，项目目的、输入输出、文件作用、运行方式、未来扩展路径都没有统一说明。
- 当前没有“轻量可运行版本”和“研究扩展版本”的明确边界，导致既不够简洁，也不够规范。

本次重构的目标是建立一个统一接口下的双模式框架：

- 当前可以用模拟数据完整演示“已知切削参数和振动，预测表面粗糙度及频域特征”的流程。
- 未来拿到真实实验数据后，可以直接接入，不需要推翻整体架构。
- 同时提供：
  - `lite` 轻量版：少依赖、好理解、方便老师快速运行和演示。
  - `research` 完整版：适合后续做真实数据、扩展模型、绘图与课题化分析。

## 2. 任务边界

### 2.1 当前正式任务

当前版本的正式监督任务定义为：

- 输入：切削参数 + 三向振动特征
- 输出：
  - 表面粗糙度 `Ra`
  - 表面起伏频域特征（默认 8 维频带或谐波特征）

对应问题描述为：

> 已知铣削切削参数、刀具刃数和加工时机床三向振动信息，希望预测切削后的表面粗糙度，同时预测表面起伏在频域中的分布特征。

### 2.2 未来扩展任务

未来版本预留以下扩展：

- 表面轮廓序列预测
- 从表面轮廓自动转换频域特征
- 从原始振动时序自动提取特征
- 支持更丰富的切削参数和工艺条件
- 支持多模型对比与消融实验

当前版本不会直接实现“轮廓序列预测模型”，但架构必须为其预留目标接口。

## 3. 设计原则

1. 统一数据接口，不让模型代码感知原始数据来源差异。
2. 轻量版和完整版共用同一套字段规范、特征定义和预测接口。
3. 明确区分“模拟数据演示能力”和“真实实验数据接入能力”。
4. 尽量保留旧脚本供对照，避免重构后失去原始作业痕迹。
5. README 以机械加工老师能快速理解为目标，而不是只对编程者友好。

## 4. 目标目录结构

重构后目录设计如下：

```text
BPNN/
├── README.md
├── requirements-lite.txt
├── requirements-research.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── outputs/
│   ├── lite/
│   └── research/
├── examples/
│   ├── sample_input.csv
│   └── sample_real_data_template.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── data_io.py
│   ├── feature_engineering.py
│   ├── targets.py
│   ├── synthetic_data.py
│   ├── models_lite.py
│   ├── models_research.py
│   ├── train_lite.py
│   ├── train_research.py
│   ├── predict.py
│   └── evaluate.py
└── legacy/
    ├── bpnn.py
    ├── generate_data.py
    └── milling_prediction_framework.py
```

### 4.1 保留旧脚本的原因

旧脚本移动到 `legacy/` 后仍可运行，用于：

- 对照“原始作业版本”和“重构规范版本”的区别。
- 让老师快速看到当前重构不是另起炉灶，而是在原有思路上工程化。
- 后续 README 中可以引用旧脚本说明演化路径。

## 5. 数据接口设计

项目必须支持两种输入入口，但内部统一到相同格式。

### 5.1 入口 A：汇总表模式

适用于“每次实验一行记录”的数据。

建议模板文件：

- `examples/sample_real_data_template.csv`

建议最小字段集合：

- 基础字段
  - `sample_id`
  - `spindle_speed_rpm`
  - `feed_per_tooth_mm`
  - `axial_depth_mm`
  - `radial_depth_mm`
  - `tool_flutes`
- 振动特征字段
  - `vib_x_rms`
  - `vib_x_std`
  - `vib_x_peak`
  - `vib_x_pp`
  - `vib_y_rms`
  - `vib_y_std`
  - `vib_y_peak`
  - `vib_y_pp`
  - `vib_z_rms`
  - `vib_z_std`
  - `vib_z_peak`
  - `vib_z_pp`
- 标签字段
  - `Ra_um`
  - `freq_bin_1` 到 `freq_bin_8`

扩展字段允许存在，例如：

- `vib_x_kurtosis`
- `vib_x_skewness`
- `sampling_rate_hz`
- `surface_profile_file`

原则是：必填字段不足时报错，可选字段存在则自动纳入。

### 5.2 入口 B：原始实验文件夹模式

适用于未来每次实验有独立原始文件的情况。

建议结构：

```text
data/raw/
├── experiment_index.csv
├── vibration/
│   ├── S001_vib_x.csv
│   ├── S001_vib_y.csv
│   ├── S001_vib_z.csv
│   └── ...
├── surface_profile/
│   ├── S001_profile.csv
│   └── ...
└── roughness/
    ├── S001_roughness.json
    └── ...
```

`experiment_index.csv` 中记录每个样本的试验条件及文件映射关系，例如：

- `sample_id`
- `spindle_speed_rpm`
- `feed_per_tooth_mm`
- `axial_depth_mm`
- `radial_depth_mm`
- `tool_flutes`
- `sampling_rate_hz`
- `vib_x_file`
- `vib_y_file`
- `vib_z_file`
- `profile_file`
- `Ra_um`

### 5.3 统一内部格式

无论输入方式是什么，预处理结束后都统一输出到：

- `data/processed/features.csv`
- `data/processed/targets.csv`

其中：

- `features.csv` 仅包含模型输入特征
- `targets.csv` 仅包含监督目标

这样训练和预测代码只依赖统一格式，不依赖原始数据组织方式。

## 6. 特征工程设计

### 6.1 输入特征层次

输入特征分三层：

1. 原始工艺参数
   - 主轴转速
   - 每齿进给量
   - 轴向切深
   - 径向切深
   - 刀具刃数
2. 派生工艺参数
   - 齿通过频率
   - 主轴频率
   - 进给速度
3. 振动特征
   - 三向 RMS、STD、Peak、PP
   - 若有原始信号可进一步提取偏度、峭度、带通能量等

### 6.2 当前默认特征集

为了兼顾理解难度与效果，当前默认最小特征集采用：

- 5 个切削基础参数
- 3 个派生参数
- 三向振动各 4 个统计量

总输入维度默认约为 20。

### 6.3 未来扩展

未来 `feature_engineering.py` 负责：

- 从三向原始振动时序自动提取统计特征
- 从频谱中提取分带能量
- 允许通过配置切换特征组合

## 7. 目标定义设计

### 7.1 当前目标层

当前默认预测目标为两类：

- `Ra_um`
- `freq_bin_1` 到 `freq_bin_8`

频域目标的物理解释可根据数据来源不同采用两种含义：

- 若来自模拟数据，可表示表面起伏谐波或归一化频带幅值。
- 若来自真实表面测量，可表示轮廓信号转换后的频带能量或指定频域统计量。

README 中需要明确说明：频域目标是“表面起伏在频域中的表征”，而非振动信号本身的频域。

### 7.2 未来目标扩展层

在 `targets.py` 中预留：

- `build_ra_targets()`
- `build_frequency_targets()`
- `build_profile_targets()`

其中 `build_profile_targets()` 在本轮重构中只保留接口定义与数据说明，不纳入正式训练流程。

## 8. 双模式模型设计

### 8.1 Lite 模式

`lite` 模式面向老师快速运行和教学演示，采用：

- 依赖：`numpy`、`pandas`、`scikit-learn`
- 模型：
  - `MLPRegressor` 或自定义轻量 BPNN
  - 多输出回归器预测频域特征
- 输出：
  - 训练指标
  - 预测结果 CSV
  - 简单 summary 文本
  - 可选图表

Lite 模式强调：

- 安装简单
- 运行稳定
- 结果足够直观

### 8.2 Research 模式

`research` 模式面向课题扩展，采用：

- 依赖：`PyTorch`、`matplotlib`、`scikit-learn`
- 模型：共享编码器 + 双任务预测头
  - `Ra` 预测头
  - 频域特征预测头
- 训练能力：
  - 训练/验证/测试划分
  - 早停
  - 最优权重保存
  - 损失历史记录
  - 图表输出

Research 模式强调：

- 面向未来真实数据
- 面向更完整实验流程
- 代码结构优先于“单文件能跑”

### 8.3 两种模式的关系

两种模式不是两套互相无关的项目，而是：

- 同一套数据接口
- 同一套字段规范
- 同一套预测任务
- 不同复杂度的模型与训练实现

## 9. 统一命令行与 Python 接口

### 9.1 数据准备命令

```bash
python -m src.data_io prepare --input <path> --output data/processed/
```

功能：

- 读取汇总表或索引表
- 检查字段
- 自动补充派生参数
- 输出标准化后的 `features.csv` 与 `targets.csv`
- 生成数据质量报告

### 9.2 Lite 训练命令

```bash
python -m src.train_lite --features data/processed/features.csv --targets data/processed/targets.csv --out outputs/lite/
```

### 9.3 Research 训练命令

```bash
python -m src.train_research --features data/processed/features.csv --targets data/processed/targets.csv --out outputs/research/
```

### 9.4 预测命令

```bash
python -m src.predict --model <model_path> --input examples/sample_input.csv --out <prediction_path>
```

支持：

- 单样本预测
- 多样本批量预测

输出统一为：

- `Ra_um`
- `freq_bin_1` 到 `freq_bin_8`
- `model_type`
- `task_level`

### 9.5 评估命令

```bash
python -m src.evaluate --model <model_path> --features <features.csv> --targets <targets.csv> --out <dir>
```

生成：

- 指标汇总
- `Ra` 对比结果
- 频域特征对比结果
- summary 文本
- 在 research 模式下额外生成图表

### 9.6 Python API

统一提供一个高层接口，例如：

```python
from src.predict import SurfaceQualityPredictor

predictor = SurfaceQualityPredictor.load("outputs/lite/model.pkl")
result = predictor.predict_from_features({...})
```

该接口的价值是：

- 以后写论文、脚本、界面时更容易复用
- 不要求调用者理解内部模型细节

## 10. README 设计要求

`README.md` 需要面向机械类老师和后续使用者，必须包含以下内容：

1. 项目背景
2. 能解决什么问题
3. 输入输出分别是什么
4. 当前版本与未来扩展的关系
5. 目录说明
6. 数据格式说明
7. 运行方式
8. 输出结果说明
9. 模拟数据与真实数据的区别
10. 常见问题

README 的核心目标是让非软件背景读者能够回答以下问题：

- 这个项目是干什么的
- 现在是否能运行
- 将来真实数据来了如何接
- 哪些结果是模拟演示，哪些接口是真实预留

## 11. 迁移策略

### 11.1 保留与迁移

- 现有 `bpnn.py`、`generate_data.py`、`milling_prediction_framework.py` 移动到 `legacy/`
- 现有 CSV 结果文件保留或迁移到 `outputs/legacy/`，避免污染新结构
- 新的入口全部放在 `src/`

### 11.2 新旧职责映射

- `generate_data.py` 的可复用部分迁移到 `src/synthetic_data.py`
- `bpnn.py` 的简化思路迁移到 `src/models_lite.py`
- `milling_prediction_framework.py` 的多任务模型与可视化思路迁移到：
  - `src/models_research.py`
  - `src/train_research.py`
  - `src/evaluate.py`

## 12. 验证要求

重构完成后至少验证：

1. `lite` 模式可基于模拟数据完整训练、预测、输出结果
2. `research` 模式可基于模拟数据完整训练、评估、输出图表
3. 使用 `examples/sample_real_data_template.csv` 能通过数据校验
4. `predict` 命令可对单行输入生成 `Ra` 和频域预测结果
5. README 中示例命令与实际代码一致

## 13. 风险与非目标

### 13.1 主要风险

- 当前没有真实实验数据，真实效果无法被本次重构直接验证
- 频域目标定义若未来测量方式变化，字段命名和计算方法可能需要调整
- 轮廓序列预测虽然预留接口，但短期内不会形成稳定任务

### 13.2 非目标

本次不追求：

- 做成完整 GUI 软件
- 做成高度自动化论文实验平台
- 在无真实数据前过度优化模型精度

本次更重要的是：

- 把逻辑理顺
- 把接口定清
- 把 README 写明白
- 把未来接实测数据的路径搭好

## 14. 实施顺序

建议的实现顺序为：

1. 建立新目录和统一字段规范
2. 迁移旧脚本到 `legacy/`
3. 实现模拟数据与真实数据模板接入
4. 实现 `lite` 训练与预测
5. 实现 `research` 训练与评估
6. 产出详尽 README
7. 运行验证并整理输出目录

## 15. 当前已知限制

- 当前项目目录不是 Git 仓库，因此本设计文档无法按规范提交版本控制。
- 本文档作为当前重构实现的唯一设计基线，后续实现需遵循本文结构。

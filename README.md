# 微震信号分类 / Microseismic Signal Classification

基于时域与时频域双分支网络的微震信号三分类研究代码，类别为岩石破裂、爆破和噪声。当前公开入口为 `classify_V4.py`。

## 公开范围与运行限制

本仓库是部分源代码发布。自定义网络层模块 **`Class/CLASS.py` 暂未公开**，`NetWork/microseismic_model.py` 依赖该模块，因此当前公开版本不能独立构建模型或完成训练。具备该模块的使用者可将其放入 `Class/CLASS.py` 后使用训练入口。该文件已加入忽略规则。

不包含训练权重、训练数据、原始样本、实验输出、历史版本、Grad-CAM 可视化脚本、预警平台或降噪模型。数据预处理脚本可独立于未公开的网络层模块运行。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `classify_V4.py` | 训练入口、训练参数及数据集划分配置 |
| `NetWork/microseismic_model.py` | 双分支网络结构 |
| `LoadData/h5_data.py` | H5 数据读取、划分与 TensorFlow 数据管道 |
| `Chart/chart.py` | 训练监控、评估和结果绘图 |
| `ST_1.3.py` | TXT 波形预处理、S 变换与 H5 生成 |

## 环境

参考环境为 Python 3.9、TensorFlow 2.17.0、Keras 3.10.0。建议使用独立环境：

```bash
python -m pip install -r requirements.txt
```

依赖安装并不补齐未公开的 `Class/CLASS.py`。

## 数据预处理

输入为单列数值 TXT 波形，采样率 4000 Hz；项目使用的速度信号单位为 m/s。代码统一长度至 4000 点，执行去均值、10–1500 Hz 带通滤波和去趋势，生成标准化时域数据及三通道时频特征。短记录补零、长记录截断的具体行为见脚本。

通过环境变量选择输入与输出目录，例如 Bash：

```bash
CLASSIFY_INPUT_DIR=data/raw/noise CLASSIFY_PREPROCESS_OUTPUT_DIR=data/processed/noise python ST_1.3.py
```

默认处理 `data/raw/noise`，输出至 `data/processed/noise`。类别标注方式及其他处理选项见脚本；处理其他类别时请检查标注结果。

## 训练（需要未公开模块）

补齐 `Class/CLASS.py` 并准备 H5 数据后：

```bash
CLASSIFY_DATA_DIR=data/processed CLASSIFY_OUTPUT_DIR=. python classify_V4.py
```

`CLASSIFY_DATA_DIR` 默认为 `data/processed`；`CLASSIFY_OUTPUT_DIR` 默认为当前目录，训练结果写入其 `training_results/` 子目录。数据、权重和输出均不纳入版本控制。

## 发布说明

从当前研究目录选择最新模块建立独立仓库；仅将个人绝对路径替换为环境变量配置并添加标准包初始化文件，保留原模型计算和训练逻辑。发布检查包括 Python 语法及文件范围检查；因自定义层未公开，未声称本仓库可独立完成训练或复现实验结果。

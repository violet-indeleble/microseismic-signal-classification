# 微震信号分类 / Microseismic Signal Classification

基于时域与时频域双分支网络的微震信号三分类研究代码，类别为岩石破裂、爆破和噪声。当前公开入口为 `classify_V4.py`。

## 公开范围与运行限制

本仓库公开当前分类模型的训练入口、网络结构、自定义层、数据读取、评估与预处理代码，包括 `Class/CLASS.py`。训练需要自行准备符合格式的 H5 数据并安装依赖。

不包含训练权重、训练数据、原始样本、实验输出、历史版本、Grad-CAM 可视化脚本、预警平台或降噪模型。数据预处理脚本可独立运行。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `classify_V4.py` | 训练入口、训练参数及数据集划分配置 |
| `NetWork/microseismic_model.py` | 双分支网络结构 |
| `Class/CLASS.py` | 自定义注意力层、多尺度卷积及特征融合模块 |
| `LoadData/h5_data.py` | H5 数据读取、划分与 TensorFlow 数据管道 |
| `Chart/chart.py` | 训练监控、评估和结果绘图 |
| `ST_1.3.py` | TXT 波形预处理、S 变换与 H5 生成 |

## 环境

参考环境为 Python 3.9、TensorFlow 2.17.0、Keras 3.10.0。建议使用独立环境：

```bash
python -m pip install -r requirements.txt
```

自定义网络层已随仓库提供。

## 数据预处理

输入为单列数值 TXT 波形，采样率 4000 Hz；项目使用的速度信号单位为 m/s。代码统一长度至 4000 点，执行去均值、10–1500 Hz 带通滤波和去趋势，生成标准化时域数据及三通道时频特征。短记录补零、长记录截断的具体行为见脚本。

通过环境变量选择输入与输出目录，例如 Bash：

```bash
CLASSIFY_INPUT_DIR=data/raw/noise CLASSIFY_PREPROCESS_OUTPUT_DIR=data/processed/noise python ST_1.3.py
```

默认处理 `data/raw/noise`，输出至 `data/processed/noise`。类别标注方式及其他处理选项见脚本；处理其他类别时请检查标注结果。

## 训练

准备 H5 数据后：

```bash
CLASSIFY_DATA_DIR=data/processed CLASSIFY_OUTPUT_DIR=. python classify_V4.py
```

`CLASSIFY_DATA_DIR` 默认为 `data/processed`；`CLASSIFY_OUTPUT_DIR` 默认为当前目录，训练结果写入其 `training_results/` 子目录。数据、权重和输出均不纳入版本控制。

## 发布说明

从当前研究目录选择最新模块建立独立仓库；仅将个人绝对路径替换为环境变量配置并添加标准包初始化文件，保留原模型计算和训练逻辑。发布检查包括 Python 语法及文件范围检查；本次发布未执行完整训练，不代表已复现实验结果。

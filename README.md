# ResNet\-CIFAR10: 基于 ResNet\-18 的 CIFAR\-10 图像分类

本项目基于 PyTorch 深度学习框架，完整复现了 ResNet\-18 残差网络在 CIFAR\-10 数据集上的图像分类任务。项目包含数据增强、模型搭建、分阶段训练、测试评估与结果可视化全流程，修复了进度条输出乱序问题，将训练轮数扩展至 60 轮，最终模型在测试集上达到 83\.18% 的最高分类准确率，可作为深度学习入门与课程实验的参考实现。

## Project Structure

```Plain Text
resnet_cifar10/
├── main.py                  # 主程序：数据加载、ResNet-18模型定义、训练测试、结果可视化
├── best_resnet_cifar10.pth  # 训练生成的最佳模型权重（测试准确率最高）
├── training_result.png      # 训练损失与准确率变化曲线图
├── result.txt               # 训练过程控制台完整输出日志
├── data/                    # 数据集存放目录
│   └── cifar-10-batches-py/ # CIFAR-10 原始数据集文件
└── README.md                # 项目说明文档
```

## Installation

### Prerequisites

推荐使用 Anaconda 管理 Python 环境，避免依赖冲突。

- Miniconda / Anaconda

- Python \>= 3\.8, \< 3\.12

- 支持 CUDA 的 NVIDIA 显卡（可选，无 GPU 时程序自动切换至 CPU 训练）

### Requirements

|Package|Purpose|
|---|---|
|torch \>= 1\.10|深度学习核心框架|
|torchvision \>= 0\.11|数据集加载、ResNet 模型与数据变换工具|
|tqdm|训练进度条可视化|
|matplotlib|训练曲线绘制|
|numpy|数值计算|

### 环境配置步骤

```bash
# 1. 创建并激活虚拟环境
conda create -n resnet python=3.9 -y
conda activate resnet

# 2. 安装PyTorch（根据自身CUDA版本选择对应安装命令，以下为CPU版示例）
pip install torch torchvision

# 3. 安装其余依赖
pip install tqdm matplotlib numpy
```

## Dataset Preparation

### 数据集说明

本项目使用 CIFAR\-10 公开基准数据集，该数据集包含 60000 张 32×32 像素的彩色图像，共分为飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车 10 个类别，其中训练集 50000 张，测试集 10000 张，各类别样本数量均衡。

### 数据集放置

1. 将下载并解压后的`cifar-10-batches-py`文件夹放入项目根目录的`data`文件夹中，路径结构与项目结构说明一致。

2. 若本地无数据集，可修改`main.py`中`torchvision.datasets.CIFAR10`的`download`参数为`True`，程序启动后会自动下载数据集至`./data`目录。

## Training

在项目根目录下执行以下命令启动训练：

```bash
python main.py
```

### 训练超参数

|参数名称|默认值|说明|
|---|---|---|
|BATCH\_SIZE|128|训练批次大小，平衡显存占用与训练速度|
|EPOCHS|60|总训练迭代轮数|
|LEARNING\_RATE|0\.01|初始学习率|
|优化器|SGD|带动量 0\.9，权重衰减 5e\-4|
|学习率调度|StepLR|每 20 轮学习率衰减为原值的 0\.1 倍，分三阶段训练|
|num\_workers|0|数据加载进程数，Windows 环境下保持 0 可避免多进程报错|

### 运行输出

- 程序自动检测可用设备，优先使用 GPU 加速训练；

- 控制台实时输出每轮训练、测试的平均损失与分类准确率，进度条显示训练进度；

- 自动保存测试准确率最高的模型权重为`best_resnet_cifar10.pth`；

- 训练全部完成后，自动生成损失与准确率曲线图并保存为`training_result.png`。

## Experiment Results

经过 60 轮完整迭代训练，模型性能指标如下：

- 最高测试准确率：**83\.18%**（第 55 轮）

- 最终训练准确率：90\.77%

- 训练过程分为三个阶段：前 20 轮快速收敛，中间 20 轮性能跃升，后 20 轮平稳精修，分阶段学习率衰减策略有效提升了模型最终精度。

## License

本项目采用 MIT 开源许可证。

> （注：部分内容可能由 AI 生成）

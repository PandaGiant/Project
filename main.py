import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys  # 新增：统一输出流，修复进度条乱序

# -------------------------- 1. 基础配置 --------------------------
# 自动选择设备：有GPU用GPU，无GPU用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")

# 超参数配置
BATCH_SIZE = 128    # 显存不足可改小为 64 或 32
EPOCHS = 60         # 修改：训练总轮数从30增加到60
LEARNING_RATE = 0.01

# -------------------------- 2. 数据加载（本地读取，关闭自动下载） --------------------------
print("正在加载本地CIFAR-10数据集...")

# 训练集数据增强 + 归一化
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机裁剪+填充，数据增强
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.ToTensor(),
    # CIFAR-10 数据集官方统计的均值与标准差
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 测试集仅做归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 加载本地数据集，download=False 直接读取本地文件
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=test_transform
)

# 数据加载器：Windows 系统设置 num_workers=0 避免多进程报错
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"训练集样本数：{len(train_dataset)} | 测试集样本数：{len(test_dataset)}")

# -------------------------- 3. 模型定义 --------------------------
# 使用 ResNet-18 模型，从零开始训练
model = torchvision.models.resnet18(weights=None)
# 修改最后一层全连接层，适配 CIFAR-10 的 10 分类任务
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)
model = model.to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
# 修改：学习率衰减步长改为20轮，匹配60轮总训练，每20轮衰减为原来的0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# -------------------------- 4. 训练与测试函数 --------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # 修复：添加 file=sys.stdout，与print同流输出，解决进度条乱序问题
    pbar = tqdm(loader, desc="训练中", file=sys.stdout)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # 前向传播 + 反向传播 + 参数更新
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计损失与准确率
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 实时更新进度条信息
        pbar.set_postfix({
            "平均损失": f"{total_loss / (batch_idx + 1):.4f}",
            "训练准确率": f"{100. * correct / total:.2f}%"
        })

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # 修复：添加 file=sys.stdout，与print同流输出
        pbar = tqdm(loader, desc="测试中", file=sys.stdout)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "平均损失": f"{total_loss / (batch_idx + 1):.4f}",
                "测试准确率": f"{100. * correct / total:.2f}%"
            })

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# -------------------------- 5. 主训练流程 --------------------------
if __name__ == '__main__':
    # 用于记录每轮数据，后续画图
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    best_acc = 0.0

    print(f"\n开始训练，总轮数：{EPOCHS}")
    for epoch in range(EPOCHS):
        print(f"\n===== 第 {epoch + 1}/{EPOCHS} 轮 =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device)

        # 记录历史数据
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_resnet_cifar10.pth")
            print(f"✅ 最佳模型已更新，当前最高测试准确率：{best_acc:.2f}%")

    print(f"\n🎉 训练全部完成！最高测试准确率：{best_acc:.2f}%")

    # -------------------------- 6. 绘制结果曲线 --------------------------
    plt.figure(figsize=(12, 5))

    # 左图：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(test_loss_history, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(test_acc_history, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_result.png", dpi=300)
    print("📊 训练结果图已保存为 training_result.png")
    plt.show()
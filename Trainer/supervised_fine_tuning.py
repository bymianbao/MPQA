import time
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
from torch.utils.data import random_split, DataLoader
import cv2
from model.ViT_B import ViT_B
from model.EfficientNet import EfficientNet_Based
from Criteria.Loss import RankLoss
from Criteria.Loss_with_cls import RankLoss_Cls
from dataset.SupervisedDataset import SupervisedDataset
from util.tools import predict

torch.manual_seed(42)

# 使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义超参数
num_epochs = 100
learning_rate = 0.0005
batch_size = 4
embedding_dim = 4  # 与Res_Based模型中的embedding_dim一致
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize([384, 384]),
                                 ])

# 训练数据集
# 加载数据集，这里使用图像数据对
# 数据标签的CSV文件
pairwise_path = r'../data/Labeled_data_train.csv'
# 文件的第一行为描述，第二行到最后一行为数据对
pairwise_data = pd.read_csv(pairwise_path, header=None)
# 获取第二行到最后一行的数据
pairwise_data = pairwise_data.iloc[:, :]


# 去除第一行后，总行数的0.8为训练集，0.2为测试集
train_size = int(len(pairwise_data) * 0.8)
val_size = len(pairwise_data) - train_size
# 训练集和验证集的划分，前train_size行为训练集，其余行为测试集
train_data, val_data = pairwise_data[:train_size], pairwise_data[train_size:]
# train_data, val_data = random_split(pairwise_data, [train_size, val_size])
# train_data的每一行，第一列为img1的图像路径，第二列为img2的图像路径，第三列为标签1，第四列为标签2，第五列为标签3，第六列为标签4
# 创建Dataset实例
train_dataset = SupervisedDataset(train_data, transform=transforms)
val_dataset = SupervisedDataset(val_data, transform=transforms)

# 创建DataLoader实例
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader=list(train_loader)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_loader=list(val_loader)
# 模型的定义
pretrained_model = "checkpoint/tf_efficientnet_b5"
model_name = "checkpoint/tf_efficientnet_b5_ft"
if not os.path.exists(model_name):
    os.makedirs(model_name)
model = ViT_B().to(device)
# model = EfficientNet_Based(structure='vit_base_patch16_384', embedding_dim=embedding_dim,).to(device)
# model.load_state_dict(torch.load(pretrained_model + '/best_model.pth'))


# 定义损失函数和优化器
epsilon = 0.1
criteria = RankLoss_Cls(epsilon=epsilon)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

best_val_loss = 1000000.0
# 训练模型
for epoch in range(num_epochs):  # 在每个epoch中，遍历训练集
    running_loss = 0.0
    model.train()
    start = time.time()
    for i, (img1, img2, dim_label1, dim_label2, labels) in enumerate(train_loader):
        # 将数据移到GPU上
        img1, img2, dim_label1, dim_label2, labels = img1.to(device), img2.to(device), dim_label1.to(device), dim_label2.to(device), labels.to(device)
        optimizer.zero_grad()
        # 前向传播
        output1, output2 = model(img1), model(img2)
        loss = criteria(output1, output2, dim_label1, dim_label2, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("Training: Epoch: {}, Batch: {}/{}, Loss: {}".format(epoch + 1, i + 1, len(train_loader), loss.item()))
    epoch_loss = running_loss / len(train_loader)
    print("***Training Loss: {}, Epoch: {}/{}***".format(epoch_loss, epoch + 1, num_epochs))
    torch.save(model, model_name + f"/model_epoch_{epoch + 1}.pth")
    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, (img1, img2, dim_label1, dim_label2, labels) in enumerate(train_loader):
            # 将数据移到GPU上
            img1, img2, dim_label1, dim_label2, labels = img1.to(device), img2.to(device), dim_label1.to(device), dim_label2.to(device), labels.to(device)
            optimizer.zero_grad()
            # 前向传播
            output1, output2 = model(img1), model(img2)
            loss = criteria(output1, output2, dim_label1, dim_label2, labels)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        print("***Validation Loss: {}, Epoch: {}/{}***".format(val_loss, epoch + 1, num_epochs))
        # 保存最优化模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Best Valid Loss: {}, Epoch: {}/{}".format(best_val_loss, epoch + 1, num_epochs))
            torch.save(model, model_name + "/best_model.pth")


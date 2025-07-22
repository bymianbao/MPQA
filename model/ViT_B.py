import torch
import torch.nn as nn
import timm
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViT_B(nn.Module):
    def __init__(self, structure='vit_base_patch16_384', embedding_dim=4):
        super(ViT_B, self).__init__()

        self.backbone = timm.create_model(structure, pretrained=True,num_classes=0, pretrained_cfg_overlay=dict(file='../Pretrain_model/' + structure + '.npz')).to(device)
        # pretrained_cfg_overlay = dict(file='../Pretrain_model/' + structure + '.pth')).to(device)
        # 加载模型参数
        # self.backbone.load_state_dict(torch.load('Pretrain_model/' + structure + '.pth'))
        # print(self.backbone)
        # print(torchsummary.summary(self.backbone, (3, 224, 224)))

        # 获取最后一个特征图的通道数
        self.mlp_s = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),  # 对特征图进行全局平均池化
            # nn.Flatten(),
            nn.Linear(768, 256),  # 第一个全连接层，将特征维度从768降到1024
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),  # Dropout
            nn.Linear(256, embedding_dim)  # 输出层，输出一个评分
        )
        self.mlp_w = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),  # 对特征图进行全局平均池化
            # nn.Flatten(),
            nn.Linear(768, 512),  # 第一个全连接层，将特征维度从768降到1024
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),  # Dropout
            nn.Linear(512, embedding_dim)  # 输出层，输出一个评分
        )

    def forward(self, x):
        features = self.backbone(x)
        # 通过MLP得到4维嵌入
        embedding = self.mlp_s(features)
        # 此外通过另一个分支得到
        weights = self.mlp_w(features)
        return embedding, weights


# 示例使用
if __name__ == "__main__":
    # 假设输入为两张图片，尺寸为[batch_size, 3, 224, 224]
    img1 = torch.randn(8, 3, 224, 224).to(device)
    img2 = torch.randn(8, 3, 224, 224).to(device)
    # 创建模型实例
    model = ViT_B().to(device)

    # print(model)
    # 前向传播
    embedding, weights = model(img1)
    # 打印嵌入维度
    print(embedding.shape)
    # 打印模型结构
    torchsummary.summary(model, (3, 224, 224))

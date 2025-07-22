import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ZLPRLoss(nn.Module):
    def __init__(self):
        super(ZLPRLoss, self).__init__()

    def forward(self, outputs, labels):
        """
        outputs: 模型输出，尺寸为 (batch_size, num_classes)
        labels: 标签，尺寸为 (batch_size, num_classes)，每个元素为 0 或 1
        """
        # 确保 outputs 和 labels 的尺寸一致
        assert outputs.size() == labels.size()

        # 分离正标签和负标签
        pos_mask = (labels == 1)  # 正标签掩码
        neg_mask = (labels == 0)  # 负标签掩码

        # 对正标签，计算 e^{-s_i}，再进行求和
        pos_term = torch.exp(-outputs[pos_mask]).sum()

        # 对负标签，计算 e^{s_j}，再进行求和
        neg_term = torch.exp(outputs[neg_mask]).sum()

        # 根据公式计算损失
        loss = torch.log(1 + pos_term) + torch.log(1 + neg_term)

        return loss


class RankLoss_Cls(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(RankLoss_Cls, self).__init__()
        self.epsilon = epsilon
        self.zlpr_loss = ZLPRLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output1, output2, dim_label1, dim_label2, labels): # 前者是
        """
        output1: (batch_size, embedding_dim*2)
        output2: (batch_size, embedding_dim*2)
        dim_label1: (batch_size, embedding_dim)
        dim_label2: (batch_size, embedding_dim)
        labels: (batch_size, embedding_dim+1)
        """
        output1 = torch.cat(output1, dim=1)
        output2 = torch.cat(output2, dim=1)
        # 第一部分的损失函数：一个多分类类别损失ZLPRLoss：
        zlpr1 = self.zlpr_loss(output1[:, 4:8], dim_label1)
        zlpr2 = self.zlpr_loss(output2[:, 4:8], dim_label2)
        zlpr_loss = (zlpr1 + zlpr2) / 2

        # 第二部分的损失函数，排序损失
        # 求总体评价指标，output1的前四个维度乘后四个维度并求和，并和前四个维度组合为一个向量
        # 计算前四个维度与后四个维度的加权和，并将其转换为二维张量以便拼接
        lambda_overall = 4
        output1[:, 4:8] = self.softmax(output1[:, 4:8])
        output2[:, 4:8] = self.softmax(output2[:, 4:8])
        sum1 = torch.sum(output1[:, :4] * output1[:, 4:8], dim=1, keepdim=True)
        out1 = torch.cat((output1[:, :4], sum1), dim=1)

        sum2 = torch.sum(output2[:, :4] * output2[:, 4:8], dim=1, keepdim=True)
        out2 = torch.cat((output2[:, :4], sum2), dim=1)
        # 计算标签为1时刻的损失
        loss1 = out2 - out1 + self.epsilon
        # 标签为1位置时,计算loss1和0的大小，取最大值
        loss1 = torch.max(loss1, torch.zeros_like(loss1))
        loss1 = torch.cat((loss1[:, :4], lambda_overall * loss1[:, 4:5]), dim=1)
        loss1 = loss1[labels == 1]

        # 计算标签为0时刻的损失
        loss2 = abs(out2 - out1)
        loss2 = torch.cat((loss2[:, :4], lambda_overall * loss2[:, 4:5]), dim=1)

        # 只在标签为0的地方取loss2的值
        loss2 = loss2[labels == 0]

        # 计算标签为-1时刻的损失
        loss3 = out1 - out2 + self.epsilon
        loss3 = torch.max(loss3, torch.zeros_like(loss3))
        loss3 = torch.cat((loss3[:, :4], lambda_overall * loss3[:, 4:5]), dim=1)
        loss3 = loss3[labels == -1]
        # 计算总损失
        # 计算总的损失
        total_loss = torch.sum(loss1) + torch.sum(loss2) + torch.sum(loss3) + zlpr_loss
        # 归一化：对 batch_size 取平均
        return total_loss / labels.size(0)


if __name__ == '__main__':
    output1 = torch.tensor(
        [[0.1, 0.8, 0.3, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5], [0.4, 0.5, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]])
    output2 = torch.tensor([[0.1, 0.8, 0.3, 1, 1, 1, 1, 0.5], [0.4, 0.5, 0.9, 1, 1, 1, 1, 1]])
    labels = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
    # 创建 RankLoss 实例
    loss_fn = RankLoss_Cls()
    # 计算损失
    loss = loss_fn(output1, output2, labels)
    print(loss.item())

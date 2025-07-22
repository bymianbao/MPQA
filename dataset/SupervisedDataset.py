from torch.utils.data import Dataset
import cv2
import torch
import os
import numpy as np
from PIL import Image

def phaser(image_path):
    dim_label = np.zeros(4)
    # 解析当image_path出现不同的字段时，使得dim的对应处置为1
    # 待解析的字段及其对应的列表
    field = {
        '-noise-': [0],
        '-contrast-': [1],
        '-crop-': [2],
        '-motion-': [3],
        '-herring-': [3],
        '-mask': [2, 3],
        '-denoise-': [0, 2]}
    for key in field.keys():
        if key in image_path:
            dim_label[field[key]] = 1
    return dim_label

class SupervisedDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.root = r'/media/jiang/hqk/project/MRI/data/LabelMRI/noisedImages'

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 读取一行的数据
        row = self.dataframe.values[idx, :]

        # 获取图像路径和标签
        img1_path = os.path.join(self.root, row[4])
        img2_path = os.path.join(self.root, row[5])
        labels = 2 - row[6:].astype(float)
        dim_label1 = phaser(img1_path)
        dim_label2 = phaser(img2_path)
        # 打开图像
        img1 = Image.open(img1_path).convert('RGB')  # 确保是 RGB 模式
        img2 = Image.open(img2_path).convert('RGB')
        # 如果定义了transform，则应用
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(dim_label1), torch.tensor(dim_label2), torch.tensor(labels)


if __name__ == '__main__':
    pass

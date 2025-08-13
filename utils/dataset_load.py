import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 自定义数据集类，用于加载 .pt 文件
class PT_Dataset(Dataset):
    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.pt_files = torch.load(data_dir)
    
    def __len__(self):

        return self.pt_files.shape[0]
    
    def __getitem__(self, index):

        return self.pt_files[index]
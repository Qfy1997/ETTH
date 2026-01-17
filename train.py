"""
cite:https://suebwj.github.io/2025/08/%E9%97%A8%E6%8E%A7%E6%AE%8B%E5%B7%AE-GRN/
Gate Residual Network
y=Gate(x)⊙F(x,W)+(−Gate(x))⊙x
时序预测方向我浏览过一些比较好的期刊，还是不能理解一些学术性比较强的东西，比如"Horizon H∈{96,192,336,720}","lookback window length"，这两个东西感觉挺难翻译的。
"""
import math
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time

torch.manual_seed(66)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MyDataset(Dataset):
   def __init__(self, train_data,train_labels):
       self.data = train_data
       self.labels=train_labels
   def __len__(self):
       return len(self.data)
   def __getitem__(self, idx):
       return self.data[idx],self.labels[idx]


class My_GRN(nn.Module):
    def __init__(self):
        super(My_GRN, self).__init__()
        self.linear1 = nn.Linear(6,6)
        self.linear2 = nn.Linear(6,1)

    def forward(self,x):
        gate = gelu(x)
        func=self.linear1(x)
        Gatefunc1=gate*func
        Gatefunc2=(-gate)*x
        res=Gatefunc1+Gatefunc2
        final_res=self.linear2(res)
        return final_res



if __name__=='__main__':
    train_data = np.load('train_data.npy')
    train_label=np.load('train_label.npy')
    train=torch.from_numpy(train_data).float()
    labels=torch.from_numpy(train_label)
    train_dataset=MyDataset(train,labels)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    model=My_GRN()
    for data, labels in dataloader:
        print(data.shape)
        print(labels.shape)
        res=model(data)
        print(res.shape)
        break
        
    

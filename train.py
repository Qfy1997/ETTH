import math
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from einops import rearrange

torch.manual_seed(66)


class MyDataset(Dataset):
   def __init__(self, train_data,train_labels):
       self.data = train_data
       self.labels=train_labels
   def __len__(self):
       return len(self.data)
   def __getitem__(self, idx):
       return self.data[idx],self.labels[idx]

class ChannelCompressAttention(nn.Module):
    def __init__(self):
        super(ChannelCompressAttention,self).__init__()
        self.conv=nn.Conv2d(6,1,kernel_size=1)
    def forward(self,x):
        res=self.conv(x)
        return res

class PileConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PileConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.A = nn.Parameter(torch.randn(6, 1024)).to("cuda:0")
        self.compress = ChannelCompressAttention().to("cuda:0")

        self.enc1=PileConv(1,64).to("cuda:0")
        self.enc2=PileConv(64,128).to("cuda:0")
        self.enc3=PileConv(128,256).to("cuda:0")
        self.enc4=PileConv(256,512).to("cuda:0")
        self.bridge=PileConv(512,1024).to("cuda:0")
        self.dec4=PileConv(1024,512).to("cuda:0")
        self.dec3=PileConv(512,256).to("cuda:0")
        self.dec2=PileConv(256,128).to("cuda:0")
        self.dec1=PileConv(128,64).to("cuda:0")
        self.upconv4=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2).to("cuda:0")
        self.upconv3=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2).to("cuda:0")
        self.upconv2=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2).to("cuda:0")
        self.upconv1=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2).to("cuda:0")
        self.pool=nn.MaxPool2d(2).to("cuda:0")
        self.l1=nn.Linear(32,1).to("cuda:0")
        self.l2 = nn.Linear(32,1).to("cuda:0")
        self.l3 = nn.Linear(64,1).to("cuda:0")

    def forward(self,x):
        x=x.unsqueeze(-1)
        res = x*self.A
        output = rearrange(res,'b n (h w) -> b n h w',w=32)
        compress_res = self.compress(output)

        enc1_out=self.enc1(compress_res)
        pool_out=self.pool(enc1_out)
        # 降维过程有结合转置卷积(ConvTranspose2d)
        enc2_out=self.enc2(pool_out)
        enc3_out=self.enc3(self.pool(enc2_out))
        enc4_out=self.enc4(self.pool(enc3_out))
        bridge_out=self.bridge(self.pool(enc4_out))
        # print("bridge out shape:",bridge_out.shape)
        dec4_out=self.dec4(torch.cat([self.upconv4(bridge_out),enc4_out],dim=1))
        dec3_out=self.dec3(torch.cat([self.upconv3(dec4_out),enc3_out],dim=1))
        dec2_out=self.dec2(torch.cat([self.upconv2(dec3_out),enc2_out],dim=1))
        dec1_out=self.dec1(torch.cat([self.upconv1(dec2_out),enc1_out],dim=1))
        # print("dec1_out shape:",dec1_out.shape)
        res1 = self.l1(dec1_out).squeeze(-1)
        res2 = self.l2(res1).squeeze(-1)
        res3 = self.l3(res2)
       
        
        return res3


if __name__=='__main__':
    train_data = np.load('train_data.npy')
    train_label=np.load('train_label.npy')
    train=torch.from_numpy(train_data).float()
    labels=torch.from_numpy(train_label).float()
    train_dataset=MyDataset(train,labels)
    dataloader = DataLoader(train_dataset, batch_size=1742, shuffle=False)
    model =My_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    print(len(dataloader))
    for epoch in range(20000):
        avg_loss=0
        start= time.time()
        for data, labels in dataloader:
            optimizer.zero_grad()
            data=data.to("cuda:0")
            labels=labels.to("cuda:0")
            res = model(data)
            loss=criterion(res,labels)
            # print("loss:",loss)
            loss.backward()
            optimizer.step()
            avg_loss+=loss
            # break
        # break
        end=time.time()
        avg_loss/=10
        print("epoch:",pre_epoch," loss:",avg_loss.detach().cpu().numpy()," time:",end-start,"s")
    torch.save(model,'mypretrain20000.pth')
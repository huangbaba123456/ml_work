# 训练集测试集

from torch.utils.data import Dataset
import torch
import numpy as np


class MyDataSet(Dataset):
    def __init__(self,train_data,seq_len,time_encode_names=["month","day","hour","week"]) -> None:
        super().__init__()
        self.time_encode=np.array(train_data[time_encode_names])
        other_columns=train_data[  [cur for cur in train_data.columns.to_list() if cur not in time_encode_names] ]
        self.train_y=np.array(other_columns)
        self.train_x=np.array(other_columns)
        self.seq_len=seq_len
    def __getitem__(self, index):
        x=torch.from_numpy(self.train_x[index:index+self.seq_len]).to(torch.float32) # (seq_len,15)
        y=torch.from_numpy(self.train_y[index+self.seq_len:index+self.seq_len*2]).to(torch.float32) # (seq_len,7)
        # y_time_encode=torch.from_numpy(self.time_encode[index+self.seq_len:index+self.seq_len*2]).to(torch.float32) # (seq_len,8)
        x_time_encode=torch.from_numpy(self.time_encode[index:index+self.seq_len]).long()
        y_time_encode=torch.from_numpy(self.time_encode[index+self.seq_len:index+self.seq_len*2]).long()
        return x,y,x_time_encode,y_time_encode
    def __len__(self):
        return len(self.train_x)-self.seq_len*2



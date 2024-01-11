# 定义模型
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
# class Time_encode(nn.Module):
#     def __init__(self,input_dim,time_encode_dim,time_range=(12,31,24,7),dropout_rate=0.0):
#         super().__init__()
#         """
#             month    11.000000
#             day      30.000000
#             hour     23.000000
#             week      6.000000
#         """
#         self.time_range=time_range
#         self.vector_table=nn.ModuleList(
#         )
#         self.drop=nn.Dropout(dropout_rate)
#         for cur in time_range:
#             self.vector_table.append(
#                 nn.Embedding(
#                     num_embeddings=cur,embedding_dim=time_encode_dim
#                 )
#             )
#         self.project=nn.Linear(input_dim,time_encode_dim)
#     def forward(self,x,x_time_encode,is_sample=True):
#         x=self.project(x) # (bs,seq_len,time_encode_dim)
#         if not is_sample:
#             # (bs,seq_len,input_dim)  (bs,seq_len,time_count)
#             x_time_encode=[
#                 self.vector_table[i](x_time_encode[:,:,i]) for i in range(len(self.vector_table))
#             ] #x_time_encode[j].shape = (bs,seq_len,time_encode_dim)
#             x_time_encode.append(x)
#             x=sum(x_time_encode)  
#         else:
#              # (bs,input_dim)(bs,time_count)
#             x_time_encode=[
#                 self.vector_table[i](x_time_encode[:,i]) for i in range(len(self.vector_table))
#             ] #  x_time_encode[j].shape= (bs,time_encode_dim)
#             x_time_encode.append(x)
#             x=sum(x_time_encode)
#         x=self.drop(x)
#         return  x
class Time_encode(nn.Module):
    def __init__(self,time_encode_dim,time_range=(12,31,24,7),dropout_rate=0.0):
        super().__init__()
        """
            month    11.000000
            day      30.000000
            hour     23.000000
            week      6.000000
        """
        self.time_range=time_range
        self.vector_table=nn.ModuleList(
        )
        self.drop=nn.Dropout(dropout_rate)
        for cur in time_range:
            self.vector_table.append(
                nn.Embedding(
                    num_embeddings=cur,embedding_dim=time_encode_dim
                )
            )
        
    def forward(self,x,x_time_encode,is_sample=True):
        if not is_sample:
            # (bs,seq_len,input_dim)  (bs,seq_len,time_count)
            x_time_encode=[
                self.vector_table[i](x_time_encode[:,:,i]) for i in range(len(self.vector_table))
            ] #x_time_encode[j].shape = (bs,seq_len,input_dim)
            # x_time_encode.append(x)
            # x=sum(x_time_encode)
             
        else:
             # (bs,input_dim)(bs,time_count)
            x_time_encode=[
                self.vector_table[i](x_time_encode[:,i]) for i in range(len(self.vector_table))
            ] #  x_time_encode[j].shape= (bs,input_dim)
            # x_time_encode.append(x)
        x_time_encode=sum(x_time_encode) # (bs,input_dim)
        x=torch.cat([x,x_time_encode],dim=-1)
        x=self.drop(x)
        return x # (bs,seq_len,input_dim+time_encode_dim) 或者 (bs,input_dim+time_encode_dim)
#         return  x

class My_lstm(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim
                 ,use_time=True
                 ,output_dim=7
                 , num_layers=3
                 , predict_len=96
                , forecast_in_one=False
                 ,dropout_rate=0.0
                 ,dropout_rate_of_time_encode=0.0
                 ,time_encode_dim=3
                 ):
        super(My_lstm, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_dim
        self.dropout_rate = dropout_rate
        self.predict_len=predict_len
        self.forecast_in_one=forecast_in_one
        self.output_dim=output_dim
        # 创建多个 LSTM 层
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTMCell(
                        (input_dim+time_encode_dim if use_time else input_dim) if i == 0 else hidden_dim, hidden_dim
                ) for i in range(num_layers)
            ]
        )
        if not forecast_in_one:
            self.fc=nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim,output_dim)
            )
        else:
            self.fc=nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim,output_dim*predict_len)
            )
        # 时间编码层
        # self.time_encode_mlp=nn.Sequential(
        #     nn.Linear(time_encode_dim,time_encode_dim*4),nn.GELU(),nn.Linear(time_encode_dim*4 ,time_encode_dim)
        # )
        self.time_encode_Table=Time_encode(time_encode_dim,time_range=(12,31,24,7),dropout_rate=dropout_rate_of_time_encode) if use_time else None
    def forward(self, x, y,x_time_encode,y_time_encode,init_states=None,user_ground=False):
        # x.shape = (bs,seq_len,input_dim)  , y.shape = (bs,seq_len,output_dim)  , 
        #  x_time_encode.shape = (bs,seq_len,time_count) y_time_encode.shape = (bs,seq_len,time_count)
        # user_ground 预测后96天时，是否使用真实数据
        x=self.time_encode_Table(x,x_time_encode,is_sample=False) if self.time_encode_Table else x
        if init_states is None:
            h_t = [torch.zeros(x.size(0), self.hidden_size, requires_grad=True).to(x.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(x.size(0), self.hidden_size, requires_grad=True).to(x.device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = init_states
        # 对于输入序列中的每个时间步进行操作
        for t in range(x.size(1)):  
            h_t[0], c_t[0] = self.lstm_layers[0](x[:, t, :], (h_t[0], c_t[0]))
            for i in range(1, self.num_layers):
                # 应用 dropout
                h_t[i - 1] = F.dropout(h_t[i - 1], p=self.dropout_rate, training=self.training)
                h_t[i], c_t[i] = self.lstm_layers[i](h_t[i - 1], (h_t[i], c_t[i]))
        if self.forecast_in_one:
            # 一口气全部预测
            input_hidden=h_t[-1] # (bs,hidden_dim)
            y_predict=self.fc(input_hidden) # (bs,output_dim*predict_len)
            y_predict=y_predict.reshape(-1,self.predict_len,self.output_dim)
            return  y_predict
        y_predict=[]
        # 我们首先预测利用x预测第一天的
        # h_t[-1].shape = (bs,hidden_size) y_time_encode[:,0,:].shape = (bs,time_encode_dim)
        input_hidden=h_t[-1] #torch.cat([h_t[-1],self.time_encode_mlp(y_time_encode[:,0,:])],dim=1) # (bs,hidden_size+time_encode_dim)
        y_predict.append(self.fc(input_hidden))
        for t in range(1,self.predict_len):
            if user_ground:
                # # 使用前一时刻的真实数据，以及前一时刻的位置编码
                # """
                #     为了减少误差传播，使用教师强制的方法将偶尔使用预测的值
                # """
                # # y[:,t-1,:].shape =y_predict[-1]= (bs,output_dim)
                # bs=y.shape[0]
                # mask=torch.bernoulli(torch.ones_like(y[:,t-1,:])*p).to(x.device) # (bs,1) mask[i][j] 的值以 p 的概率 为 1 否则为 0
                # temp=mask*y[:,t-1,:]+(1-mask)*y_predict[-1]  # 其实也就是前一时刻每一个特征，我们以 p 的概率使用真实数据，(1-p)的概率使用预测值
                # input_x=torch.cat([temp],dim=1) # (bs,input_dim)
                input_x=y[:,t-1,:]
            else:
                # 使用前一时刻的预测数据，
                input_x=y_predict[-1] # (bs,input_dim)
            # y_time_encode.shape = (bs,seq_len,time_count)

            # (bs,input_dim)(bs,time_count)
            input_x=self.time_encode_Table(
                x=input_x,x_time_encode=y_time_encode[:,t-1,:],is_sample=True
            )  if self.time_encode_Table else input_x
            # input_x=self.time_encode_Table(input_x,y_time_encode[:,t-1,:])
            # 然后过rnn
            #  h_t[0], c_t[0] = self.lstm_layers[0](x[:, t, :], (h_t[0], c_t[0]))
            for i in range(self.num_layers):
                h_t[i],c_t[i]=self.lstm_layers[i](input_x if i==0 else h_t[i-1],(h_t[i],c_t[i]))
            # 过完取最后一个隐藏层拼接 上 要预测的那个时间点的时间编码进行预测
            y_predict.append(self.fc(h_t[-1]))
        y_predict=list(map(lambda x: x.unsqueeze(1),y_predict))
        y_predict=torch.cat(y_predict,dim=1) # (bs,seq_len,output_dim)
        # 返回结果
        return y_predict
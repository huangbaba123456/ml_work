from accelerate import Accelerator
from tqdm import tqdm
from ema_pytorch import EMA
import torch.optim as optim
import os
import matplotlib.pyplot as plt 
from MYdataSet import MyDataSet
from model import My_lstm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import  MinMaxScaler
import numpy as np
class Trainer:
    def __init__(
                self
                ,model:My_lstm
                ,train_data
                ,valid_data
                ,test_data
                ,seq_len=96
                ,train_batch=4
                ,valid_batch=16
                ,train_lr=1e-3
                ,train_epochs=100 
                ,clip_grad_norm_value=1.0
                ,ema_decay = 0.995
                ,ema_update_every=10 # 每多少步跟新一次Ema
                ,ema_update_after_step=100 # 模型训练多少次后开始使用Ema跟新
                ,saveModel_and_sample_epoch = 1 # 几个轮次取保存一次，加跑验证集
                ,split_batches = True # 加速器的参数
                ,results_folder="./result" # 在训练过程中生成的图片在哪里看
                ,adam_betas=(0.9,0.99) # 优化器的参数
                ,mixed_precision = 'fp16' # 模型训练的参数
                ,ot_standardScaler=None
                ,delta=1.0
                ) -> None:
        super().__init__()
        self.accelerator=Accelerator(
            split_batches=split_batches,mixed_precision=mixed_precision
        ) #加速器
        self.device=self.accelerator.device
        self.saveModel_and_sample_epoch=saveModel_and_sample_epoch
        self.results_folder=results_folder
        self.train_epochs=train_epochs
        self.clip_grad_norm_value=clip_grad_norm_value
        self.ot_standardScaler=ot_standardScaler
        # self.pro_arr=pro_arr
        train_DataSet=MyDataSet(train_data,seq_len=seq_len)
        valid_DataSet=MyDataSet(valid_data,seq_len=seq_len)
        test_DataSet=MyDataSet(test_data,seq_len=seq_len)
        self.train_DataLoader=DataLoader(dataset=train_DataSet,batch_size=train_batch,shuffle=True)
        self.valid_DataLoader=DataLoader(dataset=valid_DataSet,batch_size=valid_batch,shuffle=False)
        self.test_DataLoader=DataLoader(dataset=test_DataSet,batch_size=valid_batch,shuffle=False)
        self.train_DataLoader,self.valid_DataLoader,self.test_DataLoader=self.accelerator.prepare(self.train_DataLoader),self.accelerator.prepare(self.valid_DataLoader),self.accelerator.prepare(self.test_DataLoader)
        if self.accelerator.is_main_process:
            self.ema_model=EMA(
                            model=model,beta=ema_decay
                            ,update_every=ema_update_every
                            ,update_after_step=ema_update_after_step
                            )
            self.ema_model=self.ema_model.to(self.device)
        # 保存的结果文件夹
        self.results_folder=results_folder
        os.makedirs(self.results_folder,exist_ok=True)      
        # 优化器
        self.optimizer=optim.Adam(
            params=model.parameters(),lr=train_lr,betas=adam_betas
        )
        self.model,self.optimizer=self.accelerator.prepare(
            model,self.optimizer
        )
        self.cur_epochs=0
        self.loss_fn=nn.MSELoss()#torch.nn.HuberLoss(reduction='mean',delta=delta)
        self.mse_loss=nn.MSELoss()
        self.mae_loss=nn.L1Loss()
        self.train_loss=None
        self.valid_loss=None
        self.valid_loss_use_groundTruth=None
        self.test_loss = None
        self.test_loss_use_groundTruth=None


    def save(self,count):
        if not self.accelerator.is_local_main_process:
            return
        data={
            "cur_epochs":self.cur_epochs
            ,
            "model":self.accelerator.get_state_dict(self.model)
            ,
            "optimizer":self.optimizer.state_dict()
            ,
            "ema_model":self.ema_model.state_dict()
            ,
            "train_loss":self.train_loss
            ,
            "valid_loss":self.valid_loss
            ,
            "valid_loss_use_groundTruth":self.valid_loss_use_groundTruth
            ,
            "test_loss":self.test_loss
            ,
            "test_loss_use_groundTruth":self.test_loss_use_groundTruth
            ,
            "ot_standardScaler":self.ot_standardScaler
            ,
            "scaler":self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None
        }
        print(f"save pth  my cur_step{self.cur_epochs}")
        torch.save(data,os.path.join(
            self.results_folder,f"model_{count}.pth"
        ))
    def load(self,ckpt):
        device=self.accelerator.device
        data=torch.load(ckpt,map_location=device)
        model=self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])
        self.cur_step=data["cur_epochs"]
        self.optimizer.load_state_dict(data["optimizer"])
        if self.accelerator.is_main_process:
            # 如果是主进程的化
            self.ema_model.load_state_dict(data["ema_model"])
        if self.accelerator.scaler is not None and data["scaler"] is not None:
            self.accelerator.scaler.load_state_dict(data["scaler"])
        self.train_loss=data["train_loss"]
        self.valid_loss=data["valid_loss"]
        self.valid_loss_use_groundTruth=data["valid_loss_use_groundTruth"]
        self.test_loss=data["test_loss"]
        self.test_loss_use_groundTruth=data["test_loss_use_groundTruth"]
        self.ot_standardScaler=data["ot_standardScaler"]
    def train_one_epoch(self):
        self.model.train()
        train_tqdm = tqdm(self.train_DataLoader, desc=f"train:{self.cur_epochs}/{self.train_epochs}")
        total_loss_mse, total_loss_mae, total_loss_my = 0, 0, 0
        total_T_loss_mse = 0.0
        total_T_loss_mae = 0.0
        for count, data in enumerate(train_tqdm):
            x, y, x_time_encode,y_time_encode = data
            count += 1
            with self.accelerator.autocast():
                y_predict = self.model(x, y, x_time_encode,y_time_encode, user_ground=True)  # (bs,seq_len,output_dim)
                # 计算损失
                loss = self.loss_fn(y_predict, y)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.accelerator.wait_for_everyone()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accelerator.wait_for_everyone()
            # 计算损失
            with torch.no_grad():
                loss_mse = self.mse_loss(y_predict, y).detach().cpu().item()
                loss_mae = self.mae_loss(y_predict, y).detach().cpu().item()
                loss_my = loss.detach().cpu().item()
                # 最后计算一下油温的损失
                T_predict = y_predict[:, :, -1].detach().cpu()  # (bs,seq_len)
                T_ground = y[:, :, -1].detach().cpu()  # (bs,seq_len)
                T_predict=T_predict.reshape(-1,1)
                T_ground=T_ground.reshape(-1,1)
                if self.ot_standardScaler is not None:
                    T_predict=self.ot_standardScaler.inverse_transform(T_predict)#transform(T_predict)
                    T_ground=self.ot_standardScaler.inverse_transform(T_ground)
                    T_predict=torch.from_numpy(T_predict)
                    T_ground=torch.from_numpy(T_ground)
                T_loss_mse = self.mse_loss(T_predict, T_ground).item()
                T_loss_mae = self.mae_loss(T_predict,T_ground).item()
            total_T_loss_mse += T_loss_mse
            total_T_loss_mae += T_loss_mae
            total_loss_mae += loss_mae
            total_loss_mse += loss_mse
            total_loss_my += loss_my
            train_tqdm.set_postfix_str(
                f"T_mse:{'{:.{}f}'.format(total_T_loss_mse / count, 5)} "
                f"mse:{'{:.{}f}'.format(total_loss_mse / count, 5)} "
                f"mae:{'{:.{}f}'.format(total_loss_mae / count, 5)} "
                f"my_loss:{'{:.{}f}'.format(total_loss_my / count, 5)}")
            train_tqdm.update()
        # 记下来这个epoch的损失
        self.train_loss["mse"].append(total_loss_mse / count)
        self.train_loss["mae"].append(total_loss_mae / count)
        self.train_loss["myLoss"].append(total_loss_my / count)
        self.train_loss["T_mse"].append(total_T_loss_mse / count)
        self.train_loss["T_mae"].append(total_T_loss_mae / count)
        train_tqdm.close()
    def valid_ones_epoch(self,user_ground):
        self.model.eval()
        valid_tqdm = tqdm(self.valid_DataLoader, desc=f"valid_use_ground:{user_ground}:{self.cur_epochs}/{self.train_epochs}")
        total_loss_mse, total_loss_mae, total_loss_my = 0, 0, 0
        total_T_loss_mse = 0
        total_T_loss_mae = 0

        for count, data in enumerate(valid_tqdm):
            x, y, x_time_encode,y_time_encode = data
            count += 1
            with torch.no_grad():
                with self.accelerator.autocast():
                    y_predict = self.model(x, y, x_time_encode,y_time_encode, user_ground=user_ground)  # (bs,seq_len,output_dim)
                    # 计算损失
                    loss = self.loss_fn(y_predict, y)
                # 计算损失
                loss_mse = self.mse_loss(y_predict, y).detach().cpu().item()
                loss_mae = self.mae_loss(y_predict, y).detach().cpu().item()
                loss_my = loss.detach().cpu().item()
                T_predict = y_predict[:, :, -1].detach().cpu()  # (bs,seq_len)
                T_ground = y[:, :, -1].detach().cpu()  # (bs,seq_len)
                T_predict=T_predict.reshape(-1,1)
                T_ground=T_ground.reshape(-1,1)
                if self.ot_standardScaler is not None:
                    T_predict=self.ot_standardScaler.inverse_transform(T_predict)
                    T_ground=self.ot_standardScaler.inverse_transform(T_ground)
                    T_predict=torch.from_numpy(T_predict)
                    T_ground=torch.from_numpy(T_ground)
                T_loss_mse = self.mse_loss(T_predict, T_ground).item()
                T_loss_mae = self.mae_loss(T_predict, T_ground).item() 
                total_T_loss_mse += T_loss_mse
                total_T_loss_mae += T_loss_mae
                total_loss_mae += loss_mae
                total_loss_mse += loss_mse
                total_loss_my += loss_my
                valid_tqdm.set_postfix_str(
                    f"T_mse:{'{:.{}f}'.format(total_T_loss_mse / count, 5)} mse:{'{:.{}f}'.format(total_loss_mse / count, 5)} mae:{'{:.{}f}'.format(total_loss_mae / count, 5)} my_loss:{'{:.{}f}'.format(total_loss_my / count, 5)}")
                valid_tqdm.update()
        valid_tqdm.close()
        if user_ground:
            self.valid_loss_use_groundTruth["mse"].append(total_loss_mse / count)
            self.valid_loss_use_groundTruth["mae"].append(total_loss_mae / count)
            self.valid_loss_use_groundTruth["myLoss"].append(total_loss_my / count)
            self.valid_loss_use_groundTruth["T_mse"].append(total_T_loss_mse / count)
            self.valid_loss_use_groundTruth["T_mae"].append(total_T_loss_mae / count)

        else:
            self.valid_loss["mse"].append(total_loss_mse / count)
            self.valid_loss["mae"].append(total_loss_mae / count)
            self.valid_loss["myLoss"].append(total_loss_my / count)
            self.valid_loss["T_mse"].append(total_T_loss_mse / count)
            self.valid_loss["T_mae"].append(total_T_loss_mae / count)

    def test_ones_epoch(self,user_ground):
        self.model.eval()
        test_tqdm = tqdm(self.test_DataLoader, desc=f"test_use_ground:{user_ground}:{self.cur_epochs}/{self.train_epochs}")
        total_loss_mse, total_loss_mae, total_loss_my = 0, 0, 0
        total_T_loss_mse = 0
        total_T_loss_mae = 0
        for count, data in enumerate(test_tqdm):
            x, y, x_time_encode,y_time_encode = data
            count += 1
            with torch.no_grad():
                with self.accelerator.autocast():
                    y_predict = self.model(x, y, x_time_encode,y_time_encode, user_ground=user_ground)  # (bs,seq_len,output_dim)
                    # 计算损失
                    loss = self.loss_fn(y_predict, y)
                # 计算损失
                loss_mse = self.mse_loss(y_predict, y).detach().cpu().item()
                loss_mae = self.mae_loss(y_predict, y).detach().cpu().item()
                loss_my = loss.detach().cpu().item()
                T_predict = y_predict[:, :, -1].detach().cpu()  # (bs,seq_len)
                T_ground = y[:, :, -1].detach().cpu()  # (bs,seq_len)
                T_predict=T_predict.reshape(-1,1)
                T_ground=T_ground.reshape(-1,1)
                if self.ot_standardScaler is not None:
                    T_predict=self.ot_standardScaler.inverse_transform(T_predict)
                    T_ground=self.ot_standardScaler.inverse_transform(T_ground)
                    T_predict=torch.from_numpy(T_predict)
                    T_ground=torch.from_numpy(T_ground)
                T_loss_mse = self.mse_loss(T_predict, T_ground).item()
                T_loss_mae = self.mae_loss(T_predict,T_ground).item()
                total_T_loss_mse += T_loss_mse
                total_T_loss_mae += T_loss_mae
                total_loss_mae += loss_mae
                total_loss_mse += loss_mse
                total_loss_my += loss_my
                test_tqdm.set_postfix_str(
                    f"T_mse:{'{:.{}f}'.format(total_T_loss_mse / count, 5)} mse:{'{:.{}f}'.format(total_loss_mse / count, 5)} mae:{'{:.{}f}'.format(total_loss_mae / count, 5)} my_loss:{'{:.{}f}'.format(total_loss_my / count, 5)}")
                test_tqdm.update()
        test_tqdm.close()
        if user_ground:
            self.test_loss_use_groundTruth["mse"].append(total_loss_mse / count)
            self.test_loss_use_groundTruth["mae"].append(total_loss_mae / count)
            self.test_loss_use_groundTruth["myLoss"].append(total_loss_my / count)
            self.test_loss_use_groundTruth["T_mse"].append(total_T_loss_mse / count)
            self.test_loss_use_groundTruth["T_mae"].append(total_T_loss_mae / count)

        else:
            self.test_loss["mse"].append(total_loss_mse / count)
            self.test_loss["mae"].append(total_loss_mae / count)
            self.test_loss["myLoss"].append(total_loss_my / count)
            self.test_loss["T_mse"].append(total_T_loss_mse / count)
            self.test_loss["T_mae"].append(total_T_loss_mae / count)
    def train(self):
        self.train_loss={"mse":[],"mae":[],"myLoss":[],"T_mse":[],"T_mae":[]} if self.train_loss is None else self.train_loss
        self.valid_loss={"mse":[],"mae":[],"myLoss":[],"T_mse":[],"T_mae":[]} if self.valid_loss is None else self.valid_loss
        self.valid_loss_use_groundTruth={"mse":[],"mae":[],"myLoss":[],"T_mse":[],"T_mae":[]} if self.valid_loss_use_groundTruth is None else self.valid_loss_use_groundTruth
        self.test_loss={"mse":[],"mae":[],"myLoss":[],"T_mse":[],"T_mae":[]} if self.test_loss is None else self.test_loss
        self.test_loss_use_groundTruth={"mse":[],"mae":[],"myLoss":[],"T_mse":[],"T_mae":[]} if self.test_loss_use_groundTruth is None else self.test_loss_use_groundTruth
        res_epochs=self.train_epochs-self.cur_epochs
        for _ in range(res_epochs):
            self.cur_epochs+=1
            self.train_one_epoch()
            # 然后验证集报告精度
            if self.cur_epochs%self.saveModel_and_sample_epoch==0:
                self.valid_ones_epoch(user_ground=True)
                self.valid_ones_epoch(user_ground=False)
                self.test_ones_epoch(user_ground=True)
                self.test_ones_epoch(user_ground=False)
            self.save(count=self.cur_epochs)
        self.accelerator.print("training complete")
        # valid_loss_use_groundTruth=np.array(self.valid_loss_use_groundTruth)
        # idx=np.argmin(valid_loss_use_groundTruth)
        names=["mse","mae","T_mse","T_mae"]
        return self.cal_epoch_loss(names=names)
    def cal_epoch_loss(self,names=["mse","mae","T_mse","T_mae"]):
        train_loss=np.array([self.train_loss[key] for key in self.train_loss.keys() if key in names]).T # (epochs,4)
        valid_loss=np.array([self.valid_loss[key] for key in self.valid_loss.keys() if key in names]).T # (epochs,4)
        test_loss=np.array([self.test_loss[key] for key in self.test_loss.keys() if key in names]).T    # (epochs,4)

        valid_loss_use_groundTruth=np.array([self.valid_loss_use_groundTruth[key] for key in self.valid_loss_use_groundTruth.keys() if key in names]).T # (epochs,4)
        test_loss_use_groundTruth=np.array([self.test_loss_use_groundTruth[key] for key in self.test_loss_use_groundTruth.keys() if key in names]).T    # (epochs,4)


        min_loss_epochs=np.argmin(valid_loss_use_groundTruth[:,0])


        train_loss=train_loss[min_loss_epochs]
        valid_loss=valid_loss[min_loss_epochs]
        test_loss=test_loss[min_loss_epochs]
        valid_loss_use_groundTruth=valid_loss_use_groundTruth[min_loss_epochs]
        test_loss_use_groundTruth=test_loss_use_groundTruth[min_loss_epochs]
        return list(train_loss),list(valid_loss),list(valid_loss_use_groundTruth),list(test_loss),list(test_loss_use_groundTruth),names

    def inference(self,use_ema_model=False,use_test=True):
        # 跑测试集
        dataDataloader=self.test_DataLoader if use_test else self.valid_DataLoader
        test_tqdm=tqdm(dataDataloader,desc=f"inference")
        model=self.model if not use_ema_model else self.ema_model.ema_model
        model.eval()
        with torch.no_grad():
            total_loss_mse,total_loss_mae,total_loss_my=0,0,0
            total_T_loss_mse=0
            for count,data in enumerate(test_tqdm):
                x, y, x_time_encode,y_time_encode = data
                count+=1
                with self.accelerator.autocast():
                    y_predict=model(x, y,x_time_encode,y_time_encode,user_ground=False)  # (bs,seq_len,output_dim)
                    # 计算损失
                    loss=self.loss_fn(y_predict,y)
                    # 计算损失
                loss_mse=self.mse_loss(y_predict,y).detach().cpu().item()
                loss_mae=self.mae_loss(y_predict,y).detach().cpu().item()
                loss_my=loss.detach().cpu().item()
                T_predict = y_predict[:, :, -1]  # (bs,seq_len)
                T_ground = y[:, :, -1]  # (bs,seq_len)
                T_loss = self.mse_loss(T_predict, T_ground).detach().cpu().item()
                total_T_loss_mse += T_loss
                total_loss_mae+=loss_mae
                total_loss_mse+=loss_mse
                total_loss_my+=loss_my
                test_tqdm.set_postfix_str(f"T_mse:{'{:.{}f}'.format(total_T_loss_mse/count, 5)} mse:{'{:.{}f}'.format(total_loss_mse/count, 5)} mae:{'{:.{}f}'.format(total_loss_mae/count, 5)} my_loss:{'{:.{}f}'.format(total_loss_my/count, 5)}")
                test_tqdm.update()
            test_tqdm.close()
        self.accelerator.print(f"T_mse:{'{:.{}f}'.format(total_T_loss_mse/count, 5)} mse:{'{:.{}f}'.format(total_loss_mse/count, 5)} mae:{'{:.{}f}'.format(total_loss_mae/count, 5)} my_loss:{'{:.{}f}'.format(total_loss_my/count, 5)}")
    def plot_figure(self,x,y,x_time_encode,y_time_encode,use_ema_model=False,output_path="./output_result",feature_names=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],use_ground=False):
        # x.shape = (bs,seq_len,input_dim)  , y.shape = (bs,seq_len,output_dim)  ,  y_time_encode.shape = (bs,seq_len,time_encode_dim)
        model=self.model if not use_ema_model else self.ema_model.ema_model
        model.eval()
        with torch.no_grad():
            x,y,x_time_encode,y_time_encode=self.accelerator.prepare(x),self.accelerator.prepare(y),self.accelerator.prepare(x_time_encode),self.accelerator.prepare(y_time_encode)
            y_predict=model(x, y,x_time_encode,y_time_encode,user_ground=use_ground)  # (1,seq_len,output_dim)
            y_predict=y_predict.transpose(-1,-2) # (bs,dim,seq_len)
        y_predict=y_predict.detach().cpu().numpy() # (bs,output_dim,seq_len)
        y=y.transpose(-1,-2) # (bs,output_dim,seq_len)
        x=x.transpose(-1,-2) # (bs,input_dim,seq_len)
        x=x[:,:len(feature_names),:]  # (bs,output_dim,seq_len)
        # print(x.shape,y.shape)
        y=torch.cat([x,y],dim=-1).detach().cpu().numpy() # (bs,output_dim,seq_len*2)
        assert y.shape[1]==y_predict.shape[1]==len(feature_names)
        # 画图
        for cur_batch in [0,1]:#range(x.shape[0]):
            os.makedirs(f"{output_path}/cur_batch:{cur_batch}",exist_ok=True)
            cur_batch_y,cur_batch_y_predict=y[cur_batch],y_predict[cur_batch]
            for i in range(cur_batch_y.shape[0]):
                plt.figure(figsize=(30,5))
                index=torch.arange(cur_batch_y.shape[1]).numpy()
                plt.plot(index,cur_batch_y[i],label="GroundTruth")
                plt.plot(index[-cur_batch_y_predict.shape[1]:],cur_batch_y_predict[i],label="predict_value")
                plt.legend()
                plt.savefig(os.path.join(f"{output_path}/cur_batch:{cur_batch}",f"{feature_names[i]}.png"))
                plt.close()
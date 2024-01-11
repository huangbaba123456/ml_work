from utils import  time_encode,split_train_valid_test,feature_Normalization,cos_increase
import pandas as pd
from model import  My_lstm
from Trainer import Trainer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import  numpy as np
import warnings
from tqdm import tqdm

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    iters=5
    data = pd.read_csv("./ETTh1.csv")
    data = time_encode(data)

    train_data, valid_data, test_data = split_train_valid_test(data, ratios=[6, 2, 2])
    columns_names = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    train_data, standardScaler = feature_Normalization(train_data, columns_names, standardScaler=None)
    valid_data, _ = feature_Normalization(valid_data, columns_names, standardScaler=standardScaler)
    test_data, _ = feature_Normalization(test_data, columns_names=columns_names, standardScaler=standardScaler)
    # print("****10 \n\n")
    ot_standardScaler=StandardScaler()
    ot_standardScaler.fit(train_data[["OT"]])
    train_data.loc[:,"OT"]=ot_standardScaler.transform(train_data[["OT"]])
    valid_data.loc[:,"OT"]=ot_standardScaler.transform(valid_data[["OT"]])
    test_data.loc[:,"OT"]=ot_standardScaler.transform(test_data[["OT"]])

    print(train_data.shape, valid_data.shape, test_data.shape)
    ans={"train_loss":[],"valid_loss":[],"valid_loss_use_groundTruth":[],"test_loss":[],"test_loss_use_groundTruth":[]}
    seq_len = 96
    train_batch = 32*2
    valid_batch = train_batch*4
    train_epochs=10
    
    # model=My_lstm(
    #         input_dim=7, hidden_dim=256
    #         , use_time=False
    #         , output_dim=7
    #         , num_layers=4
    #         , predict_len=seq_len
    #         , forecast_in_one=False
    #         , dropout_rate=0.7
    #         , dropout_rate_of_time_encode=0.15
    #         , time_encode_dim=3
    #     )
    model=My_lstm(
            input_dim=7, hidden_dim=1024
            , use_time=False
            , output_dim=7
            , num_layers=3
            , predict_len=seq_len
            , forecast_in_one=False
            , dropout_rate=0.7
            , dropout_rate_of_time_encode=0.15
            , time_encode_dim=3
        )

    trainer=Trainer(
            model=model
            , train_data=train_data
            , valid_data=valid_data
            , test_data=test_data
            , seq_len = seq_len
            , train_batch = train_batch
            , valid_batch = valid_batch
            , train_lr = 1e-3
            , train_epochs = train_epochs
            , clip_grad_norm_value = 1.0
            , ema_decay = 0.995
            , ema_update_every = 10  # 每多少步跟新一次Ema
            , ema_update_after_step = 100  # 模型训练多少次后开始使用Ema跟新
            , saveModel_and_sample_epoch = 1  # 几个轮次取保存一次，加跑验证集
            , split_batches = True  # 加速器的参数
            , results_folder = f"./result_no_use_time/iter"  # 在训练过程中生成的图片在哪里看
            , adam_betas = (0.9, 0.99)  # 优化器的参数
            , mixed_precision = 'fp16'  # 模型训练的参数
            # ,  pro_arr=pro_arr
            ,ot_standardScaler=ot_standardScaler
            ,delta=2.0
        )
    trainer.load(
        f"./result_no_use_time{seq_len}/iter_1/model_10.pth"
    )
    print(len(trainer.valid_loss_use_groundTruth["mse"]))
    idx=np.array(trainer.valid_loss_use_groundTruth["mse"]).argmin()
    print(idx)
    idx=np.array(trainer.test_loss_use_groundTruth["mse"]).argmin()
    print(idx)
    trainer.load(
        f"./result_no_use_time{seq_len}/iter_1/model_{idx+1}.pth"
    )
    test_DataLoader=trainer.test_DataLoader
    test_DataLoader=tqdm(test_DataLoader,desc="test")
    count=0
    for data in test_DataLoader:
        x, y, x_time_encode,y_time_encode = data
        trainer.plot_figure(x,y,x_time_encode,y_time_encode,use_ema_model=False,output_path=f"./test{seq_len}:{False}/{count}",use_ground=False)
        count+=1
    
    test_DataLoader=tqdm(test_DataLoader,desc="test")
    count=0
    for data in test_DataLoader:
        x, y, x_time_encode,y_time_encode = data
        trainer.plot_figure(x,y,x_time_encode,y_time_encode,use_ema_model=False,output_path=f"./test{seq_len}:{True}/{count}",use_ground=True)
        count+=1
    

    

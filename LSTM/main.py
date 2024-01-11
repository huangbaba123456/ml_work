from utils import  time_encode,split_train_valid_test,feature_Normalization,cos_increase
import pandas as pd
from model import  My_lstm
from Trainer import Trainer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import  numpy as np
import warnings
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
    for e in range(iters):
        model=My_lstm(
            input_dim=7, hidden_dim=512
            , use_time=False
            , output_dim=7
            , num_layers=5
            , predict_len=seq_len
            , forecast_in_one=True
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
            , results_folder = f"./result_ones_predict{seq_len}/iter_{e+1}"  # 在训练过程中生成的图片在哪里看
            , adam_betas = (0.9, 0.99)  # 优化器的参数
            , mixed_precision = 'fp16'  # 模型训练的参数
            # ,  pro_arr=pro_arr
            ,ot_standardScaler=ot_standardScaler
            ,delta=2.0
        )
        # self.train_loss[idx],self.valid_loss[idx],self.valid_loss_use_groundTruth[idx],self.test_loss[idx],self.test_loss_use_groundTruth[idx]
        train_loss,valid_loss,valid_loss_use_groundTruth,test_loss,test_loss_use_groundTruth,names=trainer.train()
        ans["train_loss"].append(train_loss)
        ans["valid_loss"].append(valid_loss)
        ans["valid_loss_use_groundTruth"].append(valid_loss_use_groundTruth)
        ans["test_loss"].append(test_loss)
        ans["test_loss_use_groundTruth"].append(test_loss_use_groundTruth)
    for key in ans.keys():
        ans[key]=np.array(ans[key]) # (iters,4)
        ans[key]=(ans[key].mean(axis=0),ans[key].std(axis=0)) # (4,)
    for key in ans.keys():
        print(key)
        for i,cur_name in enumerate(names):
            print(f"{cur_name}:mean={ans[key][0][i]} std={ans[key][1][i]}")
        print() 
    # print(f"train loss:{ans[0]} valid loss:{ans[1]} valid loss use groundTruth:{ans[2]} test loss:{ans[3]} valid loss use groundTruth:{ans[4]}")



import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import math

def cos_increase(p_min=0.5,p_max=0.98,epochs=100,):
    def cosine_annealing(step, total_steps, lr_start, lr_end):
        return lr_start + 0.5 * (lr_end - lr_start) * (1 + math.cos(math.pi * step / total_steps))
    pro_arr=[]
    for step in range(epochs):
        cur_p = cosine_annealing(step, epochs, p_min, p_max)
        pro_arr.append(cur_p)
    pro_arr=pro_arr[::-1]
    return (pro_arr)

def split_data(date_str):
    year_month_day=date_str.split(" ")[0]
    year=int(year_month_day.split("-")[0])
    month=int(year_month_day.split("-")[1])
    day=int(year_month_day.split("-")[-1])
    count=day_of_week(year,month,day)
    hour=int(date_str.split(" ")[-1].split(":")[0])
    return month,day,hour,count
def sin_cos_encode(data,max_value=None):
    if max_value is None:
        max_value=data.max()
    sin_value,cos_value=np.sin(2*np.pi*data/max_value),np.cos(2*np.pi*data/max_value)
    return sin_value,cos_value
from sklearn.preprocessing import MinMaxScaler
def feature_Normalization(data,columns_names,standardScaler=None):
    if standardScaler is None:
        standardScaler=StandardScaler()
        standardScaler.fit(data[columns_names])
    data[columns_names]=standardScaler.transform(data[columns_names])
    return data,standardScaler
def time_encode(data):
    temp_arr=[]
    for i in range(len(data["date"])):
        temp_arr.append(
            split_data(date_str=data["date"][i])
        )
    temp_arr=np.array(temp_arr)
    # 时间编码
    data["month"]=(temp_arr[:,0])-1
    data["day"]=(temp_arr[:,1])-1
    data["hour"]=(temp_arr[:,2])
    # count_arr=[]
    # # ["month_sin","month_cos","day_sin","day_cos","hour_sin","hour_cos","week_sin","week_cos"]
    # for cur in temp_arr[:,-1]:
    #     count_arr.append(cur)
    # count_arr=np.array(count_arr)
    data["week"]=temp_arr[:,3]
    del data["date"]
    return data
def day_of_week(year, month, day):
    date = datetime(year, month, day)
    # Return the day name, e.g., "Monday", "Tuesday", etc.
    value_map={"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
    return value_map[date.strftime("%A")]


def split_train_valid_test(data,ratios=[6,2,2]):
    ratios=np.array(ratios)
    ratios=ratios/np.sum(ratios)
    train_ratio,valid_ratio=ratios[0],ratios[1]
    data_len=len(data)
    train_len,valid_len=int(data_len*train_ratio),int(data_len*valid_ratio)
    train_data=data[:train_len]
    valid_data=data[train_len:train_len+valid_len]
    test_data=data[train_len+valid_len:]
    return train_data,valid_data,test_data
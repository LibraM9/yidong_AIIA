# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 2feature_label.py
# @time  : 2019/7/25
"""
文件说明：针对label做feature
存在leak D3A28F27F3D43AC986EC1F217E5E7F57
"""
import pandas as pd
# import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta

# path = "F:/项目相关/1907cm_station/data/"
# out = "F:/项目相关/1907cm_station/feature/"

path = "/home/dev/lm/cm_station/data/"
out = "/home/dev/lm/cm_station/feature/"

train1 = pd.read_csv(open(path+"训练故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
train1.columns = ["id","time","station","error"]
error_type = {"电力故障":0,"硬件故障":1,"传输故障":2,"软件故障":3,"动环故障":4,"误告警":5}
train1["error"] = train1["error"].replace(error_type)
train1["is_train"]=1
"""
{0: '电力故障', 1: '硬件故障', 2: '传输故障', 3: '软件故障', 4: '动环故障', 5: '误告警'}
"""
train2 = pd.read_csv(open(path+"训练告警.csv",encoding="gb2312"),parse_dates=["告警开始时间"])
train2.columns = ["time_alert","alert","station"]
# train2["time_alert"] = pd.to_datetime(train2["time_alert"])
test1 = pd.read_csv(open(path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
test1.columns = ["id","time","station","error"]
test1["is_train"]=0
test2 = pd.read_csv(open(path+"测试告警.csv",encoding="gb2312"),parse_dates=["告警开始时间"])
test2.columns = ["time_alert","alert","station"]

error = pd.concat([train1,test1],axis=0)
error = error.reset_index(drop=True)
alert = pd.concat([train2,test2],axis=0)
alert = alert.reset_index(drop=True)
alert = alert.drop_duplicates()

# 前5 10 30 60 分钟
error["time_last5"] = error["time"].apply(lambda x: x - relativedelta(minutes=+5))
error["time_last10"] = error["time"].apply(lambda x: x - relativedelta(minutes=+10))
error["time_last30"] = error["time"].apply(lambda x: x - relativedelta(minutes=+30))
error["time_last60"] = error["time"].apply(lambda x: x - relativedelta(minutes=+60))
error["time_future5"] = error["time"].apply(lambda x: x + relativedelta(minutes=+5))
error["time_future10"] = error["time"].apply(lambda x: x + relativedelta(minutes=+10))
error["time_future30"] = error["time"].apply(lambda x: x + relativedelta(minutes=+30))
error["time_future60"] = error["time"].apply(lambda x: x + relativedelta(minutes=+60))

error["hour"] = error["time"].apply(lambda x:str(x)[11:13])
error["day"] = error["time"].apply(lambda x:int(str(x)[8:10]))
error["day_of_week"] = error["time"].dt.dayofweek

func_list = []
for i in range(6):
    exec("""def cnt{}(df): 
            if df.count()==0 : 
                return np.nan 
            else: 
                return df[df=={}].count()""".format(i,i))
    func_list.append("cnt{}".format(i))
agg = {"error":[eval(i) for i in func_list]}

# all构造特征
print("all")
error_union = error.drop("error",axis=1).merge(error[["id","time","station","error"]].rename(columns={"id":"id1","time":"time1"}),how = 'left',on=["station"])
error_union = error_union.loc[error_union["id"]!=error_union["id1"]]
error_tmp = error_union.groupby(["station","id","is_train"], as_index=False).agg(agg)
error_tmp.columns = ['time_all_' + col[0] + '_' + col[1] for col in error_tmp.columns]
error_tmp = error_tmp.rename(columns={'time_all_station_':'station','time_all_id_':'id','time_all_is_train_':'is_train'})
error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

# last构造特征
for i in [0,5,10,30,60,"all"]:
    print(i)
    error_union = error.drop("error", axis=1).merge(error[["id", "time", "station", "error"]].rename(columns={"id": "id1", "time": "time1"}), how='left',on=["station"])
    error_union = error_union.loc[error_union["id"] != error_union["id1"]]
    if i==0:
        error_union = error_union.loc[error_union["time"]==error_union["time1"]]
    elif i == "all":
        error_union = error_union.loc[(error_union["time"] >= error_union["time1"])]
    else:
        error_union = error_union.loc[(error_union["time"] >= error_union["time1"])&(error_union["time1"]>error_union["time_last{}".format(i)])]
    error_tmp = error_union.groupby(["station","id","is_train"], as_index=False).agg(agg)
    error_tmp.columns = ['time_last{}_'.format(i) + col[0] + '_' + col[1] for col in error_tmp.columns]
    error_tmp = error_tmp.rename(columns={'time_last{}_station_'.format(i):'station','time_last{}_id_'.format(i):'id','time_last{}_is_train_'.format(i):'is_train'})
    error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

# future特征构造
for i in [5,10,30,60,"all"]:
    print(i)
    error_union = error.drop("error", axis=1).merge(error[["id", "time", "station", "error"]].rename(columns={"id": "id1", "time": "time1"}), how='left',on=["station"])
    error_union = error_union.loc[error_union["id"] != error_union["id1"]]
    if i == "all":
        error_union = error_union.loc[(error_union["time"] <= error_union["time1"])]
    else:
        error_union = error_union.loc[(error_union["time"] <= error_union["time1"])&(error_union["time1"]<error_union["time_future{}".format(i)])]
    error_tmp = error_union.groupby(["station","id","is_train"], as_index=False).agg(agg)
    error_tmp.columns = ['time_future{}_'.format(i) + col[0] + '_' + col[1] for col in error_tmp.columns]
    error_tmp = error_tmp.rename(columns={'time_future{}_station_'.format(i):'station','time_future{}_id_'.format(i):'id','time_future{}_is_train_'.format(i):'is_train'})
    error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

del error["time_last5"]
del error["time_last10"]
del error["time_last30"]
del error["time_last60"]
del error["time_future5"]
del error["time_future10"]
del error["time_future30"]
del error["time_future60"]
del error["error"],error["time"],error["hour"],error["day"],error["day_of_week"]

error.to_csv(out+"data_label0725.csv",index=False)
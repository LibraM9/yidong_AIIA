#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : error_embedding.py
# @Author: Peng
# @Date  : 2019/9/9
# @Desc  :


import embeddingandstacking
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta
import gc
#error embedding
path = '/home/internship/gp/mlcompetition/yidong/errorClass/fusai/'
out = './feature/'
train1 = pd.read_csv(open(path+"训练故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
train1.columns = ["id","station","time","error"]#60346,5
error_type={}
count = 0
n='a0'
for i in train1.error.value_counts().index:
    error_type[i]=n
    count += 1
    n='a{}'.format(count)
train1["error"] = train1["error"].replace(error_type)
train1["is_train"]=1
test1 = pd.read_csv(open(path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
test1.columns = ["id","station","time","error"]#6696,5
test1["is_train"]=0
error = pd.concat([train1,test1],axis=0)
error = error.reset_index(drop=True)
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

def listvalue(x):
    return list(x.values)
agg = {"error_list":listvalue}

#所有故障
error_union = error.drop("error",axis=1).merge(error[["id","time","station","error"]].rename(columns={"id":"id1","time":"time1"}),how = 'left',on=["station"])
error_union = error_union.loc[error_union["id"]!=error_union["id1"]]
error_union = error_union.loc[~error_union["error"].isna()]
error_union = error_union.sort_values('time1',ascending = True)
error_tmp = error_union.groupby(["id","station","is_train"], as_index=False)['error'].agg(agg)
temp = error_tmp.reset_index(drop = True)
temp_values = temp['error_list']
error_embedding_result = embeddingandstacking.embedding(temp_values, size=8)[1]
error_embedding_result = pd.DataFrame(error_embedding_result,columns = ['error_embedding_all_{}'.format(j) for j in range(8)])
temp_error = pd.concat([temp, error_embedding_result], axis=1)
del temp_error['error_list']
error = error.merge(temp_error,how = 'left',on=["id","station","is_train"])
#error last embedding
for i in [5,10,30,60,"all"]:
    print('last_error_{}'.format(i))
    error_union = error.drop("error", axis=1).merge(error[["id", "time", "station", "error"]].rename(columns={"id": "id1", "time": "time1"}), how='left',on=["station"])
    error_union = error_union.loc[error_union["id"] != error_union["id1"]]
    error_union = error_union.loc[~error_union["error"].isna()]
    if i==0:
        error_union = error_union.loc[error_union["time"]==error_union["time1"]]
    elif i == "all":
        error_union = error_union.loc[(error_union["time"] >= error_union["time1"])]
    else:
        error_union = error_union.loc[(error_union["time"] >= error_union["time1"])&(error_union["time1"]>error_union["time_last{}".format(i)])]
    error_union = error_union.sort_values('time1',ascending = True)
    error_tmp = error_union.groupby(["id","station","is_train"], as_index=False)['error'].agg(agg)
    temp = error_tmp.reset_index(drop=True)
    temp_values = temp['error_list']
    error_embedding_result = embeddingandstacking.embedding(temp_values, size=8)[1]
    error_embedding_result = pd.DataFrame(error_embedding_result,
                                          columns=['error_embedding_last_{}_{}'.format(i, j) for j in range(8)])
    temp_error = pd.concat([temp, error_embedding_result], axis=1)
    print(1)
    del temp_error['error_list']
    print(2)
    error = error.merge(temp_error, how='left', on=["id", "station", "is_train"])

#future
for i in [5,10,30,60,"all"]:
    print('future_error_{}'.format(i))
    error_union = error.drop("error", axis=1).merge(error[["id", "time", "station", "error"]].rename(columns={"id": "id1", "time": "time1"}), how='left',on=["station"])
    error_union = error_union.loc[error_union["id"] != error_union["id1"]]
    error_union = error_union.loc[~error_union["error"].isna()]
    if i == "all":
        error_union = error_union.loc[(error_union["time"] <= error_union["time1"])]
    else:
        error_union = error_union.loc[(error_union["time"] <= error_union["time1"])&(error_union["time1"]<error_union["time_future{}".format(i)])]
    error_union = error_union.sort_values('time1',ascending = True)
    error_tmp = error_union.groupby(["station","id","is_train"], as_index=False)['error'].agg(agg)
    temp = error_tmp.reset_index(drop=True)
    temp_values = temp['error_list']
    error_embedding_result = embeddingandstacking.embedding(temp_values, size=8)[1]
    error_embedding_result = pd.DataFrame(error_embedding_result,
                                          columns=['error_embedding_future_{}_{}'.format(i, j) for j in range(8)])
    temp_error = pd.concat([temp, error_embedding_result], axis=1)
    del temp_error['error_list']
    error = error.merge(temp_error, how='left', on=["id", "station", "is_train"])
del error['day']
del error['hour']
del error['day_of_week']
del error['error']
del error['time']
del error["time_last5"]
del error["time_last10"]
del error["time_last30"]
del error["time_last60"]
del error["time_future5"]
del error["time_future10"]
del error["time_future30"]
del error["time_future60"]
error.to_csv(out+'error_embedding.csv',index = False)


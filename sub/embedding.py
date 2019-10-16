#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : embedding.py
# @Author: Peng
# @Date  : 2019/9/3
# @Desc  :
import embeddingandstacking
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta
import gc
# path = "F:/项目相关/1907cm_station/data/"
# out = "F:/项目相关/1907cm_station/feature/"
path = "C:\\Users\\gupeng\\Desktop\\yidong\\errorClass\\fusai\\"
out = "C:\\Users\\gupeng\\Desktop\\yidong\\errorClass\\fusai\\feature\\"
path = '/home/dev/lm/cm_station/data/fusai/'
out = '/home/dev/lm/gupeng/feature/'
train1 = pd.read_csv(open(path+"训练故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
train1.columns = ["id","station","time","error"]#60346,5
error_type={}
n=0
for i in train1.error.value_counts().index:
    error_type[i]=n
    n=n+1
"""
{'主设备-硬件故障': 0,
 '其他-误告警或自动恢复': 1,
 '动力环境-电力部门供电': 2,
 '主设备-参数配置异常': 3,
 '主设备-设备复位问题': 4,
 '主设备-软件故障': 5,
 '动力环境-开关电源': 6,
 '主设备-设备连线故障': 7,
 '动力环境-动力环境故障': 8,
 '主设备-信源问题': 9,
 '传输系统-光缆故障': 10,
 '传输系统-其他原因': 11,
 '动力环境-高低压设备': 12,
 '动力环境-电源线路故障': 13,
 '动力环境-环境': 14,
 '传输系统-传输设备': 15,
 '动力环境-UPS': 16,
 '动力环境-动环监控系统': 17,
 '主设备-其他': 18,
 '人为操作-告警测试': 19,
 '人为操作-工程施工': 20,
 '人为操作-物业原因': 21,
 '主设备-天馈线故障': 22}
"""
train1["error"] = train1["error"].replace(error_type)
train1["is_train"]=1
train2 = pd.read_csv(open(path+"训练告警.csv",encoding="gb2312"),parse_dates=["告警发生时间"])
train2.columns = ["station","alert","time_alert"]#3401351,3 大量同一时刻2条相同数据
test1 = pd.read_csv(open(path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
test1.columns = ["id","station","time","error"]#6696,5
test1["is_train"]=0
test2 = pd.read_csv(open(path+"测试告警.csv",encoding="gb2312"),parse_dates=["告警发生时间"])
test2.columns = ["station","alert","time_alert"]#1212175,3
error = pd.concat([train1,test1],axis=0)
error = error.reset_index(drop=True)
test2 = test2.loc[~test2.station.isin(train2.station)]
alert = pd.concat([train2,test2],axis=0)
alert = alert.reset_index(drop=True)
# alert = alert.drop_duplicates()
le = preprocessing.LabelEncoder()
alert["alert"] = le.fit_transform(alert["alert"])
alert_type = {index: label for index, label in enumerate(le.classes_)}
alert_type_new = dict(zip([i for i in range(130)],['b{}'.format(j) for j in range(130)]))
alert['alert'] = alert['alert'].replace(alert_type_new)
# 前5 10 30 60 分钟
error["time_last5"] = error["time"].apply(lambda x: x - relativedelta(minutes=+5))
error["time_last10"] = error["time"].apply(lambda x: x - relativedelta(minutes=+10))
error["time_last30"] = error["time"].apply(lambda x: x - relativedelta(minutes=+30))
error["time_last60"] = error["time"].apply(lambda x: x - relativedelta(minutes=+60))
error["time_future5"] = error["time"].apply(lambda x: x + relativedelta(minutes=+5))
error["time_future10"] = error["time"].apply(lambda x: x + relativedelta(minutes=+10))
error["time_future30"] = error["time"].apply(lambda x: x + relativedelta(minutes=+30))
error["time_future60"] = error["time"].apply(lambda x: x + relativedelta(minutes=+60))
func_list = []
errortmp = error.copy(deep = True)

#统计顺序
def listvalue(x):
    return list(x.values)
agg = {"alert_list":listvalue}

#all alert embedding:
error_union = errortmp.merge(alert,how = 'left',on=["station"])
error_union = error_union.sort_values('time_alert',ascending = True)
error_tmp = error_union.groupby(["station","id","is_train"], as_index=False)['alert'].agg(agg)
temp = error_tmp.reset_index(drop = True)
temp_values = temp['alert_list']
error_embedding_result = embeddingandstacking.embedding(temp_values, size=8)[1]
error_embedding_result = pd.DataFrame(error_embedding_result,columns = ['alert_embedding_all_{}'.format(j) for j in range(8)])
temp_error = pd.concat([temp, error_embedding_result], axis=1)
del temp_error['alert_list']
error = error.merge(temp_error,how = 'left',on=["id","station","is_train"])

del error["time_last5"]
del error["time_last10"]
del error["time_last30"]
del error["time_last60"]
del error["time_future5"]
del error["time_future10"]
del error["time_future30"]
del error["time_future60"]
del error['error']
del error['time']
error.to_csv(out+'alert_embedding_all.csv',index = False)

path = "C:\\Users\\gupeng\\Desktop\\yidong\\errorClass\\fusai\\"
out = "C:\\Users\\gupeng\\Desktop\\yidong\\errorClass\\fusai\\feature\\"
path = '/home/dev/lm/cm_station/data/fusai/'
out = '/home/dev/lm/gupeng/feature/'
train1 = pd.read_csv(open(path+"训练故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
train1.columns = ["id","station","time","error"]#60346,5
error_type={}
n=0
for i in train1.error.value_counts().index:
    error_type[i]=n
    n=n+1
"""
{'主设备-硬件故障': 0,
 '其他-误告警或自动恢复': 1,
 '动力环境-电力部门供电': 2,
 '主设备-参数配置异常': 3,
 '主设备-设备复位问题': 4,
 '主设备-软件故障': 5,
 '动力环境-开关电源': 6,
 '主设备-设备连线故障': 7,
 '动力环境-动力环境故障': 8,
 '主设备-信源问题': 9,
 '传输系统-光缆故障': 10,
 '传输系统-其他原因': 11,
 '动力环境-高低压设备': 12,
 '动力环境-电源线路故障': 13,
 '动力环境-环境': 14,
 '传输系统-传输设备': 15,
 '动力环境-UPS': 16,
 '动力环境-动环监控系统': 17,
 '主设备-其他': 18,
 '人为操作-告警测试': 19,
 '人为操作-工程施工': 20,
 '人为操作-物业原因': 21,
 '主设备-天馈线故障': 22}
"""
train1["error"] = train1["error"].replace(error_type)
train1["is_train"]=1
train2 = pd.read_csv(open(path+"训练告警.csv",encoding="gb2312"),parse_dates=["告警发生时间"])
train2.columns = ["station","alert","time_alert"]#3401351,3 大量同一时刻2条相同数据
test1 = pd.read_csv(open(path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
test1.columns = ["id","station","time","error"]#6696,5
test1["is_train"]=0
test2 = pd.read_csv(open(path+"测试告警.csv",encoding="gb2312"),parse_dates=["告警发生时间"])
test2.columns = ["station","alert","time_alert"]#1212175,3
error = pd.concat([train1,test1],axis=0)
error = error.reset_index(drop=True)
test2 = test2.loc[~test2.station.isin(train2.station)]
alert = pd.concat([train2,test2],axis=0)
alert = alert.reset_index(drop=True)
# alert = alert.drop_duplicates()
le = preprocessing.LabelEncoder()
alert["alert"] = le.fit_transform(alert["alert"])
alert_type = {index: label for index, label in enumerate(le.classes_)}
alert_type_new = dict(zip([i for i in range(130)],['b{}'.format(j) for j in range(130)]))
alert['alert'] = alert['alert'].replace(alert_type_new)
# 前5 10 30 60 分钟
error["time_last5"] = error["time"].apply(lambda x: x - relativedelta(minutes=+5))
error["time_last10"] = error["time"].apply(lambda x: x - relativedelta(minutes=+10))
error["time_last30"] = error["time"].apply(lambda x: x - relativedelta(minutes=+30))
error["time_last60"] = error["time"].apply(lambda x: x - relativedelta(minutes=+60))
error["time_future5"] = error["time"].apply(lambda x: x + relativedelta(minutes=+5))
error["time_future10"] = error["time"].apply(lambda x: x + relativedelta(minutes=+10))
error["time_future30"] = error["time"].apply(lambda x: x + relativedelta(minutes=+30))
error["time_future60"] = error["time"].apply(lambda x: x + relativedelta(minutes=+60))
func_list = []
errortmp = error.copy(deep = True)

#统计顺序
def listvalue(x):
    return list(x.values)
agg = {"alert_list":listvalue}

# print(gupeng)
for i in [0,5,10,30,"all"]:
    print('last_alert_{}'.format(i))
    error_union = errortmp.merge(alert,how = 'left',on=["station"])
    if i==0:
        error_union = error_union.loc[error_union["time"]==error_union["time_alert"]]
    elif i == "all":
        error_union = error_union.loc[(error_union["time"] >= error_union["time_alert"])]
    else:
        error_union = error_union.loc[(error_union["time"] >= error_union["time_alert"])&(error_union["time_alert"]>error_union["time_last{}".format(i)])]
    #升序，前5分钟
    error_union = error_union.sort_values('time_alert',ascending = True)
    error_tmp = error_union.groupby(["id","station","is_train"], as_index=False)['alert'].agg(agg)
    temp = error_tmp.reset_index(drop = True)
    temp_values = temp['alert_list']
    error_embedding_result = embeddingandstacking.embedding(temp_values, size=8)[1]
    error_embedding_result = pd.DataFrame(error_embedding_result,columns = ['alert_embedding_last_{}_{}'.format(i,j) for j in range(8)])
    temp_error = pd.concat([temp, error_embedding_result], axis=1)
    del temp_error['alert_list']
    error = error.merge(temp_error,how = 'left',on=["id","station","is_train"])
    gc.collect()
#future embedding
for i in [5,10,30,"all"]:
    print('futurealert_{}'.format(i))
    error_union = errortmp.merge(alert,how = 'left',on=["station"])
    if i == "all":
        error_union = error_union.loc[(error_union["time"] <= error_union["time_alert"])]
    else:
        error_union = error_union.loc[(error_union["time"] <= error_union["time_alert"])&(error_union["time_alert"]<error_union["time_future{}".format(i)])]
    error_union = error_union.sort_values('time_alert', ascending=True)
    error_tmp = error_union.groupby(["id","station","is_train"], as_index=False)['alert'].agg(agg)
    temp = error_tmp.reset_index(drop=True)
    temp_values = temp['alert_list']
    error_embedding_result = embeddingandstacking.embedding(temp_values, size=8)[1]
    error_embedding_result = pd.DataFrame(error_embedding_result,
                                          columns=['alert_embedding_future_{}_{}'.format(i, j) for j in range(8)])
    temp_error = pd.concat([temp, error_embedding_result], axis=1)
    del temp_error['alert_list']
    error = error.merge(temp_error, how='left', on=["id", "station", "is_train"])
    gc.collect()

del error["time_last5"]
del error["time_last10"]
del error["time_last30"]
del error["time_last60"]
del error["time_future5"]
del error["time_future10"]
del error["time_future30"]
del error["time_future60"]
del error['error']
del error['time']
error.to_csv(out+'alert_embedding.csv',index = False)

# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 2.2embedding.py
# @time  : 2019/8/18
"""
文件说明：将告警看作时间序列，进行embedding
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta
from gensim.models import Word2Vec
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import sys
sys.path.append("/home/dev/lm/utils_lm")
# path = "F:/项目相关/1907cm_station/data/"
# out = "F:/项目相关/1907cm_station/feature/"

path = "/home/dev/lm/cm_station/data/fusai/"
out = "/home/dev/lm/cm_station/feature/fusai/"

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
alert = pd.concat([train2,test2],axis=0)
alert = alert.reset_index(drop=True)
alert = alert.drop_duplicates()

le = preprocessing.LabelEncoder()
alert["alert"] = le.fit_transform(alert["alert"])
alert_type = {index: label for index, label in enumerate(le.classes_)}

error["time_last1"] = error["time"].apply(lambda x: x - relativedelta(hours=+1))
error["time_last3"] = error["time"].apply(lambda x: x - relativedelta(hours=+3))
error["time_last6"] = error["time"].apply(lambda x: x - relativedelta(hours=+6))
error["time_last12"] = error["time"].apply(lambda x: x - relativedelta(hours=+12))
# 近若干小时的告警序列
def concat_alert(df):
    s = []
    for i in df.values:
        s.append(i)
    return s

agg = {
    "alert":[concat_alert]
}
from model_train.a1_preprocessing_nlp import embedding
for i in [1,3,6,12]:
    print(i)

    error_union = error.merge(alert, how='left', on=["station"])
    error_union = error_union.loc[(error_union["time"] >= error_union["time_alert"]) & (
            error_union["time_alert"] > error_union["time_last{}".format(i)])]
    error_union["alert"] = error_union["alert"].astype(str).apply(lambda x: "a" + x)
    error_group = error_union.groupby(["id", "is_train"], as_index=False).agg(agg)
    error_group.columns = ["id", "is_train", "alert"]

    # 构造ids 所有告警集合
    ids = sorted(list(set(error_union["alert"].values)))
    print("alert种类", len(ids))
    #embedding
    station_alert_embedding = embedding(error_group["alert"],ids,size=16)[1]
    station_alert_embedding = pd.DataFrame(station_alert_embedding, columns=['station_alert{}_em0'.format(i), 'station_alert{}_em1'.format(i),
                                                               'station_alert{}_em2'.format(i), 'station_alert{}_em3'.format(i),
                                                               'station_alert{}_em4'.format(i), 'station_alert{}_em5'.format(i),
                                                               'station_alert{}_em6'.format(i), 'station_alert{}_em7'.format(i),
                                                                             'station_alert{}_em8'.format(i),
                                                                             'station_alert{}_em9'.format(i),
                                                                             'station_alert{}_em10'.format(i),
                                                                             'station_alert{}_em11'.format(i),
                                                                             'station_alert{}_em12'.format(i),
                                                                             'station_alert{}_em13'.format(i),
                                                                             'station_alert{}_em14'.format(i),
                                                                             'station_alert{}_em15'.format(i),
                                                                             ])
    error_tmp = pd.concat([error_group[["id","is_train"]], station_alert_embedding], axis=1)
    error = error.merge(error_tmp,how = 'left',on=["id","is_train"])

del error["time"],error["station"],error["error"]
del error["time_last1"],error["time_last3"],error["time_last6"],error["time_last12"]

error.to_csv(out+"data_embedding16_0818.csv",index=False)



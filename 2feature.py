# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 2feature.py
# @time  : 2019/7/19
"""
文件说明：针对告警特征构造
工单前10分钟/30分钟/60分钟统计
GSP 频繁项挖掘
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
# le = preprocessing.LabelEncoder()
# train1["error"] = le.fit_transform(train1["error"])
# error_type = {index: label for index, label in enumerate(le.classes_)}
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

le = preprocessing.LabelEncoder()
alert["alert"] = le.fit_transform(alert["alert"])
alert_type = {index: label for index, label in enumerate(le.classes_)}
"""
{0: '4G基站退服告警',
 1: 'BBU CPRI光模块/电接口不在位告警',
 2: 'BBU CPRI光模块故障告警',
 3: 'BBU CPRI接口异常告警',
 4: 'BBU CPRI线速率协商异常告警',
 5: 'BBU IR光模块/电接口不在位告警',
 6: 'BBU IR光模块收发异常告警',
 7: 'BBU IR光模块故障告警',
 8: 'BBU IR接口异常告警',
 9: 'BBU光模块收发异常告警',
 10: 'BBU直流输出异常告警',
 11: 'License Feature不可用告警',
 12: 'MAC错帧超限告警',
 13: 'RHUB CPRI接口异常告警',
 14: 'RHUB与pRRU间链路异常告警',
 15: 'RHUB交流掉电告警',
 16: 'RHUB光模块/电接口不在位告警',
 17: 'RHUB光模块收发异常告警',
 18: 'RHUB光模块故障告警',
 19: 'RHUB时钟异常告警',
 20: 'RHUB硬件故障告警',
 21: 'S1接口故障告警',
 22: 'SCTP链路拥塞告警',
 23: 'SCTP链路故障告警',
 24: '[衍生告警]华为同一跟踪区码10分钟内出现BBU IR光模块收发异常告警合并一条',
 25: '[衍生告警]同一eNodeB的S1接口故障告警超限',
 26: '[衍生告警]同一eNodeB的传输光接口异常告警超限',
 27: '[衍生告警]同一eNodeB的射频单元IR接口异常告警超限',
 28: '[衍生告警]同一eNodeB的射频单元硬件故障告警超限',
 29: '[衍生告警]同一eNodeB的小区不可用告警超限',
 30: '[衍生告警]同一eNodeB的用户面承载链路故障告警超限',
 31: '[衍生告警]同一网元射频单元业务不可用告警超门限',
 32: '[衍生告警]同一网元射频单元光模块收发异常告警超门限',
 33: '[衍生告警]同一网元射频单元维护链路异常告警超门限',
 34: '[衍生告警]爱立信同地区多个小区E-RAB建立成功率低劣化',
 35: '主控板插错槽位告警',
 36: '交流掉电告警',
 37: '以太网链路故障告警',
 38: '传输光接口异常告警',
 39: '传输光模块不在位告警',
 40: '传输光模块故障告警',
 41: '制式间机柜配置冲突告警',
 42: '单板时钟输入异常告警',
 43: '单板温度异常告警',
 44: '单板硬件故障告警',
 45: '单板输入电压异常告警',
 46: '同一机房多个基站退服告警',
 47: '基站S1控制面传输中断告警',
 48: '射频单元ALD电流异常告警',
 49: '射频单元CPRI接口异常告警',
 50: '射频单元IR接口异常告警',
 51: '射频单元业务不可用告警',
 52: '射频单元交流掉电告警',
 53: '射频单元光模块/电接口不在位告警',
 54: '射频单元光模块收发异常告警',
 55: '射频单元光模块故障告警',
 56: '射频单元时钟异常告警',
 57: '射频单元直流掉电告警',
 58: '射频单元硬件故障告警',
 59: '射频单元维护链路异常告警',
 60: '射频单元软件运行异常告警',
 61: '射频单元输入电源能力不足告警',
 62: '射频单元运行时拓扑异常告警',
 63: '射频单元驻波告警',
 64: '小区PCI冲突告警',
 65: '小区不可用告警',
 66: '小区接收通道干扰噪声功率不平衡告警',
 67: '小区闭塞告警',
 68: '时钟参考源异常告警',
 69: '时间同步失败告警',
 70: '星卡天线故障告警',
 71: '星卡时钟输出异常告警',
 72: '星卡维护链路异常告警',
 73: '星卡锁星不足告警',
 74: '未配置时钟参考源告警',
 75: '本地用户连续登录尝试失败告警',
 76: '版本自动回退告警',
 77: '电调天线数据丢失告警',
 78: '电调天线未校准告警',
 79: '电调天线运行数据异常告警',
 80: '电调天线马达故障告警',
 81: '系统无License运行告警',
 82: '系统时钟不可用告警',
 83: '系统时钟失锁告警',
 84: '系统超出License容量限制告警',
 85: '网元连接中断',
 86: '网元遭受攻击告警',
 87: '远程维护通道配置与运行数据不一致告警'}
"""
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
for i in range(88):
    exec("def cnt{}(df): return df[df=={}].count()".format(i,i))
    func_list.append("cnt{}".format(i))
agg = {"alert":[eval(i) for i in func_list]}

# all构造特征
print("all")
error_union = error.merge(alert,how = 'left',on=["station"])
error_tmp = error_union.groupby(["station","id","is_train"], as_index=False).agg(agg)
error_tmp.columns = ['time_all_' + col[0] + '_' + col[1] for col in error_tmp.columns]
error_tmp = error_tmp.rename(columns={'time_all_station_':'station','time_all_id_':'id','time_all_is_train_':'is_train'})
error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

# last构造特征
for i in [0,5,10,30,60,"all"]:
    print(i)
    error_union = error.merge(alert,how = 'left',on=["station"])
    if i==0:
        error_union = error_union.loc[error_union["time"]==error_union["time_alert"]]
    elif i == "all":
        error_union = error_union.loc[(error_union["time"] >= error_union["time_alert"])]
    else:
        error_union = error_union.loc[(error_union["time"] >= error_union["time_alert"])&(error_union["time_alert"]>error_union["time_last{}".format(i)])]
    error_tmp = error_union.groupby(["station","id","is_train"], as_index=False).agg(agg)
    error_tmp.columns = ['time_last{}_'.format(i) + col[0] + '_' + col[1] for col in error_tmp.columns]
    error_tmp = error_tmp.rename(columns={'time_last{}_station_'.format(i):'station','time_last{}_id_'.format(i):'id','time_last{}_is_train_'.format(i):'is_train'})
    error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

# future特征构造
for i in [5,10,30,60,"all"]:
    print(i)
    error_union = error.merge(alert,how = 'left',on=["station"])
    if i == "all":
        error_union = error_union.loc[(error_union["time"] <= error_union["time_alert"])]
    else:
        error_union = error_union.loc[(error_union["time"] <= error_union["time_alert"])&(error_union["time_alert"]<error_union["time_future{}".format(i)])]
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

error.to_csv(out+"data_all0725.csv",index=False)
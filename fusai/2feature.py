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
"""
{0: 'BBU CPRI光模块/电接口不在位告警',
 1: 'BBU CPRI光模块故障告警',
 2: 'BBU CPRI接口异常告警',
 3: 'BBU CPRI线速率协商异常告警',
 4: 'BBU IR光模块/电接口不在位告警',
 5: 'BBU IR光模块收发异常告警',
 6: 'BBU IR光模块故障告警',
 7: 'BBU IR接口异常告警',
 8: 'BBU互联端口异常告警',
 9: 'BBU光模块收发异常告警',
 10: 'BBU直流输出异常告警',
 11: 'BBU风扇堵转告警',
 12: 'E1/T1环回告警',
 13: 'LTE小区E-RAB建立成功率低于三级门限（非VIP）',
 14: 'LTE小区RRC连接建立成功率低于三级门限',
 15: 'LTE小区切换成功率低于三级门限',
 16: 'LTE小区切换成功率低于三级门限（非VIP）',
 17: 'License Feature不可用告警',
 18: 'MAC错帧超限告警',
 19: 'RHUB CPRI接口异常告警',
 20: 'RHUB与pRRU间链路异常告警',
 21: 'RHUB交流掉电告警',
 22: 'RHUB光模块/电接口不在位告警',
 23: 'RHUB光模块收发异常告警',
 24: 'RHUB光模块故障告警',
 25: 'RHUB时钟异常告警',
 26: 'RHUB硬件故障告警',
 27: 'S1接口故障告警',
 28: 'SCTP链路拥塞告警',
 29: 'SCTP链路故障告警',
 30: 'Ups故障',
 31: 'VIP小区0流量信令告警（测试2）',
 32: 'VIP小区0流量信令告警（测试）',
 33: 'VIP小区0流量告警',
 34: 'VVIPLTE小区VoLTE无线接通率严重低告警',
 35: 'VVIP小区LTE0流量告警',
 36: 'VVIP小区LTE切换成功率低告警',
 37: 'VVIP小区LTE寻呼记录接收个数为0告警',
 38: 'VVIP小区LTE无线接通率低告警',
 39: 'VVIP小区LTE的PDCCH信道CCE占用率高告警',
 40: 'VVIP小区LTE的RRC用户数高告警',
 41: 'VVIP小区RRC最大连接数为0告警',
 42: '[VOLTE]LTE小区VoLTE无线接通率(语音)低于三级门限',
 43: '[衍生告警]同一基站非特殊外告告警在无退服告警时关联',
 44: '[衍生告警]同一站点跨专业关联告警',
 45: '[衍生告警]同地市同时产生多条相同告警',
 46: 'eNodeB退服告警',
 47: '中继远程节点不可用告警',
 48: '中继远程节点接收信号质量差告警',
 49: '主区版本不可用告警',
 50: '主控板插错槽位告警',
 51: '主电告警',
 52: '交流掉电告警',
 53: '以太网链路故障告警',
 54: '传输光接口异常告警',
 55: '传输光模块不在位告警',
 56: '传输光模块故障告警',
 57: '传输后备电直流过欠压告警',
 58: '低压告警',
 59: '制式间机柜配置冲突告警',
 60: '单板时钟输入异常告警',
 61: '单板温度异常告警',
 62: '单板硬件故障告警',
 63: '单板输入电压异常告警',
 64: '基站S1控制面传输中断告警',
 65: '基站控制面传输中断告警',
 66: '基站时钟失步告警',
 67: '射频单元ALD电流异常告警',
 68: '射频单元CPRI接口异常告警',
 69: '射频单元IR接口异常告警',
 70: '射频单元业务不可用告警',
 71: '射频单元交流掉电告警',
 72: '射频单元光模块/电接口不在位告警',
 73: '射频单元光模块收发异常告警',
 74: '射频单元光模块故障告警',
 75: '射频单元时钟异常告警',
 76: '射频单元直流掉电告警',
 77: '射频单元硬件故障告警',
 78: '射频单元维护链路异常告警',
 79: '射频单元软件运行异常告警',
 80: '射频单元输入电源能力不足告警',
 81: '射频单元运行时拓扑异常告警',
 82: '射频单元驻波告警',
 83: '小区不可用告警',
 84: '小区接收通道干扰噪声功率不平衡告警',
 85: '小区模拟负载启动告警',
 86: '小区闭塞告警',
 87: '整流模块告警',
 88: '时钟参考源异常告警',
 89: '时间同步失败告警',
 90: '星卡天线故障告警',
 91: '星卡时钟输出异常告警',
 92: '星卡维护链路异常告警',
 93: '星卡锁星不足告警',
 94: '普通LTE小区VoLTE无线接通率严重低告警',
 95: '普通LTE小区切换成功率严重低告警',
 96: '普通LTE小区接通率低严重低告警',
 97: '普通LTE小区无线掉线率严重高告警',
 98: '普通LTE小区的RRC用户数严重高告警',
 99: '未配置时钟参考源告警',
 100: '本地用户连续登录尝试失败告警',
 101: '水浸告警',
 102: '温度告警',
 103: '湿度告警',
 104: '烟雾告警',
 105: '熔丝告警',
 106: '版本自动回退告警',
 107: '用户面故障告警',
 108: '电源模块异常告警',
 109: '电调天线数据丢失告警',
 110: '电调天线未校准告警',
 111: '电调天线运行数据异常告警',
 112: '电调天线马达故障告警',
 113: '直流输出异常告警',
 114: '系统无License运行告警',
 115: '系统时钟不可用告警',
 116: '系统时钟失锁告警',
 117: '系统超出License容量限制告警',
 118: '网元连接中断',
 119: '网元遭受攻击告警',
 120: '蓄电池停止供电告警',
 121: '蓄电池电流异常告警',
 122: '远程维护通道配置与运行数据不一致告警',
 123: '配置文件损坏告警',
 124: '门禁告警',
 125: '门禁设备告警',
 126: '防盗告警',
 127: '雷击告警',
 128: '（用户感知）上网/语音类常驻小区投诉预警',
 129: '（监控室）VIP小区0流量信令告警（5分钟）'}
"""
# 前5 10 30 60 分钟
error["time_last2"] = error["time"].apply(lambda x: x - relativedelta(minutes=+2))
error["time_last5"] = error["time"].apply(lambda x: x - relativedelta(minutes=+5))
error["time_last10"] = error["time"].apply(lambda x: x - relativedelta(minutes=+10))
error["time_last30"] = error["time"].apply(lambda x: x - relativedelta(minutes=+30))
error["time_last60"] = error["time"].apply(lambda x: x - relativedelta(minutes=+60))

error["time_future2"] = error["time"].apply(lambda x: x + relativedelta(minutes=+2))
error["time_future5"] = error["time"].apply(lambda x: x + relativedelta(minutes=+5))
error["time_future10"] = error["time"].apply(lambda x: x + relativedelta(minutes=+10))
error["time_future30"] = error["time"].apply(lambda x: x + relativedelta(minutes=+30))
error["time_future60"] = error["time"].apply(lambda x: x + relativedelta(minutes=+60))

error["hour"] = error["time"].apply(lambda x:str(x)[11:13])
error["day"] = error["time"].apply(lambda x:int(str(x)[8:10]))
error["day_of_week"] = error["time"].dt.dayofweek

func_list = []
for i in range(130):
    exec("def cnt{}(df): return df[df=={}].count()".format(i,i))
    func_list.append("cnt{}".format(i))
agg = {"alert":[eval(i) for i in func_list]}

errortmp=error.copy(deep=True)#防止error占用内存过多
# all构造特征
print("all")
error_union = errortmp.merge(alert,how = 'left',on=["station"])
error_tmp = error_union.groupby(["station","id","is_train"], as_index=False).agg(agg)
error_tmp.columns = ['time_all_' + col[0] + '_' + col[1] for col in error_tmp.columns]
error_tmp = error_tmp.rename(columns={'time_all_station_':'station','time_all_id_':'id','time_all_is_train_':'is_train'})
error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

import gc
gc.collect()
# last构造特征
for i in [0,2,5,10,30,60,"all"]:
    print(i)
    error_union = errortmp.merge(alert,how = 'left',on=["station"])
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
    gc.collect()
# future特征构造
for i in [2,5,10,30,60,"all"]:
    print(i)
    error_union = errortmp.merge(alert,how = 'left',on=["station"])
    if i == "all":
        error_union = error_union.loc[(error_union["time"] <= error_union["time_alert"])]
    else:
        error_union = error_union.loc[(error_union["time"] <= error_union["time_alert"])&(error_union["time_alert"]<error_union["time_future{}".format(i)])]
    error_tmp = error_union.groupby(["station","id","is_train"], as_index=False).agg(agg)
    error_tmp.columns = ['time_future{}_'.format(i) + col[0] + '_' + col[1] for col in error_tmp.columns]
    error_tmp = error_tmp.rename(columns={'time_future{}_station_'.format(i):'station','time_future{}_id_'.format(i):'id','time_future{}_is_train_'.format(i):'is_train'})
    error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

del error["time_last2"]
del error["time_last5"]
del error["time_last10"]
del error["time_last30"]
del error["time_last60"]
del error["time_future2"]
del error["time_future5"]
del error["time_future10"]
del error["time_future30"]
del error["time_future60"]

error.to_csv(out+"data_all0809.csv",index=False)
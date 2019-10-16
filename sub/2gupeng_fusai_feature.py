#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gupeng_fusai_feature.py
# @Author: Peng
# @Date  : 2019/8/13
# @Desc  :


import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
# path = "F:/项目相关/1907cm_station/data/"
# out = "F:/项目相关/1907cm_station/feature/"
ori_path = '/home/internship/gp/mlcompetition/yidong/errorClass/fusai/'
path =ori_path
out = './feature/'
train1 = pd.read_csv(open(path+"训练故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
train1.columns = ["id","station","time","error"]#60346,5

group = train1.groupby('id')
x = group.apply(lambda x: x['error'].values[0].split('-')[0]).to_frame().reset_index()
y = group.apply(lambda x: x['error'].values[0].split('-')[1]).to_frame().reset_index()
x.columns = ['id','main_class']
y.columns = ['id','next_class']
train1 = train1.merge(x,on = 'id',how = 'left')
train1 = train1.merge(y,on = 'id',how = 'left')
train1["is_train"]=1
test1 = pd.read_csv(open(path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
test1.columns = ["id","station","time","error"]#6696,5
test1["main_class"]=np.nan
test1["next_class"]=np.nan
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
func_list = []
#去掉空值  是0 还是nan
for i,j in enumerate(error.next_class.value_counts().index):
    exec("""def cnt{}(df): 
            df = df.dropna()
            if df.count()==0 : 
                return np.nan
            else: 
                return (df[df=='{}'].count())/(len(df))""".format(i,str(j)))
    func_list.append("cnt{}".format(i))
agg = {"next_class1":[eval(i) for i in func_list]}
'''
error.next_class.value_counts()
硬件故障        19252
误告警或自动恢复    14320
电力部门供电       8698
参数配置异常       4794
设备复位问题       3998
软件故障         2760
开关电源         1400
设备连线故障       1365
动力环境故障        536
信源问题          495
光缆故障          428
其他原因          386
高低压设备         350
电源线路故障        259
环境            199
传输设备          193
UPS           174
动环监控系统        151
其他            140
告警测试          136
工程施工          132
物业原因           95
天馈线故障          85


'''

'''
主设备        硬件故障          19252
            参数配置异常         4794
            设备复位问题         3998
            软件故障           2760
            设备连线故障         1365
            信源问题            495
            其他              140
            天馈线故障            85
人为操作        告警测试            136
            工程施工            132
            物业原因             95
传输系统        光缆故障            428
            其他原因            386
            传输设备            193
其他          误告警或自动恢复      14320
动力环境        电力部门供电         8698
            开关电源           1400
            动力环境故障          536
            高低压设备           350
            电源线路故障          259
            环境              199
            UPS             174
            动环监控系统          151
Name: next_class, dtype: int64'''


'''


'''
# func_list1 = []
# #去掉空值  是0 还是nan
# for i,j in enumerate(error.main_class.value_counts().index):
#     exec("""def cnt{}(df):
#             df = df.dropna()
#             if df.count()==0 :
#                 return np.nan
#             else:
#                 return (df[df=='{}'].count())/(len(df))""".format(i,str(j)))
#     func_list1.append("cnt{}".format(i))
# agg1 = {"main_class1":[eval(i) for i in func_list1]}
#这里有空值怎么办？
#last特征
dic = dict(zip(train1.main_class.value_counts().index,[i for i in range(len(train1.main_class.value_counts().index))]))
for i in [0,5,10,30,60,"all"]:
    print(i)
    error_union = error.drop("error", axis=1).merge(error[["id", "time", "station", "error",'main_class','next_class']].rename(columns={"id": "id1", "time": "time1",'next_class':'next_class1','main_class':'main_class1'}), how='left',on=["station"])
    error_union = error_union.loc[error_union["id"] != error_union["id1"]]
    # error_union_tmp = error_union.copy(deep = True)
    if i==0:
        error_union = error_union.loc[error_union["time"]==error_union["time1"]]
    elif i == "all":
        error_union = error_union.loc[(error_union["time"] >= error_union["time1"])]
    else:
        error_union = error_union.loc[(error_union["time"] >= error_union["time1"])&(error_union["time1"]>error_union["time_last{}".format(i)])]
    error_union_tmp2 = error_union.copy(deep =True)
    for j in train1.main_class.value_counts().index:
        error_union = error_union_tmp2.loc[error_union_tmp2.main_class1 == j]
        error_tmp = error_union.groupby(["id","station","is_train"], as_index=False).agg(agg)
        error_tmp.columns = ["link_last_{}_{}_".format(i,dic[j]) + col[0] + '_' + col[1] for col in error_tmp.columns]
        error_tmp = error_tmp.rename(columns={"link_last_{}_{}_station_".format(i,dic[j]):'station',"link_last_{}_{}_id_".format(i,dic[j]):'id',"link_last_{}_{}_is_train_".format(i,dic[j]):'is_train'})
        error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])


#Future

for i in [5,10,30,60,"all"]:
    print(i)
    error_union = error.drop("error", axis=1).merge(error[["id", "time", "station", "error",'main_class','next_class']].rename(columns={"id": "id1", "time": "time1",'next_class':'next_class1','main_class':'main_class1'}), how='left',on=["station"])
    error_union = error_union.loc[error_union["id"] != error_union["id1"]]
    if  i == "all":
        error_union = error_union.loc[(error_union["time"] <= error_union["time1"])]
    else:
        error_union = error_union.loc[(error_union["time"] <= error_union["time1"])&(error_union["time1"]<error_union["time_last{}".format(i)])]
    error_union_tmp2 = error_union.copy(deep =True)
    for j in train1.main_class.value_counts().index:
        error_union = error_union_tmp2.loc[error_union_tmp2.main_class1 == j]
        error_tmp = error_union.groupby(["id","station","is_train"], as_index=False).agg(agg)
        error_tmp.columns = ["link_future_{}_{}_".format(i,dic[j]) + col[0] + '_' + col[1] for col in error_tmp.columns]
        error_tmp = error_tmp.rename(columns={"link_future_{}_{}_station_".format(i,dic[j]):'station',"link_future_{}_{}_id_".format(i,dic[j]):'id',"link_future_{}_{}_is_train_".format(i,dic[j]):'is_train'})
        error = error.merge(error_tmp,how = 'left',on=["id","station","is_train"])

del error['error']
del error['main_class']
del error['next_class']
del error['time']
del error["time_last5"]
del error["time_last10"]
del error["time_last30"]
del error["time_last60"]
del error["time_future5"]
del error["time_future10"]
del error["time_future30"]
del error["time_future60"]
del error["hour"],error["day"],error["day_of_week"]
error.to_csv(out+'gupeng0820.csv',index = False)

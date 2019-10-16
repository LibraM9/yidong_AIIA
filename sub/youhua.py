#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : houchuli.py
# @Author: Peng
# @Date  : 2019/8/20
# @Desc  :
import pandas as pd
import heapq
import numpy as np
s = '传输系统-传输设备、传输系统-光缆故障、传输系统-其他原因、动力环境-UPS、动力环境-电力部门供电、动力环境-电源线路故障、动力环境-动环监控系统、动力环境-动力环境故障、动力环境-高低压设备、动力环境-环境、动力环境-开关电源、其他-误告警或自动恢复、人为操作-告警测试、人为操作-工程施工、人为操作-物业原因、主设备-参数配置异常、主设备-其他、主设备-软件故障、主设备-设备复位问题、主设备-设备连线故障、主设备-天馈线故障、主设备-信源问题、主设备-硬件故障'
s = s.split('、')
column = {'主设备-硬件故障': 0,
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
raw = '/home/dev/lm/cm_station/data/fusai/'  #原始文件
ori_path = './'
out_path = ori_path + "out/"
model = 'lgb'
result = pd.read_csv(out_path+'result.csv')
best_score = pd.read_csv(out_path+'result.csv')
result1 = pd.DataFrame()
best_score1 = pd.DataFrame()
for i in (column.keys()):
    result1[i] = result[i]
    best_score1[i] = best_score[i]
result1 = result1.values
oof3 = pd.Series(list(result1))
#最大的少于0.5 第三个高于0.08 di 二者相差在0.03 之间
def find(x):
    lii = heapq.nlargest(4, x)
    if ((max(lii)<0.6) and (lii[2] > 0.07) and (lii[2] - lii[3])<0.01) :
        return 1
    else:
        return 0
def find2(x):
    lii = heapq.nlargest(4, x)
    if ((max(lii)<0.5) and (lii[2] > 0.08) and (lii[2] - lii[3])<0.01) :
        return 1
    else:
        return 0

oof_max = oof3.apply(lambda x: heapq.nlargest(4, x))
oof3_no_delete = oof3.apply(lambda x: find2(x))# 40条数据
oof3_no_delete2 = oof3.apply(lambda x: find(x))# 76条数据
#相同值怎么办
oof3_index=oof3.apply(lambda x:list(map(list(x).index, heapq.nlargest(4,x))))
oof3_unsure = oof3_index.apply(lambda x:x[2:4]).to_frame()
oof3_unsure['index0'] = oof3_unsure.apply(lambda x :x[0][0],axis = 1)
oof3_unsure['index1'] = oof3_unsure.apply(lambda x :x[0][1],axis = 1)
oof3_unsure['no_delete'] = oof3_no_delete
oof3_unsure['no_delete2'] = oof3_no_delete2
oof3_unsure = pd.concat([oof3_unsure,oof3_index,oof_max],axis = 1)
oof3_unsure_1 = oof3_unsure.loc[oof3_unsure.no_delete == 1]
oof3_unsure_2 = oof3_unsure.loc[oof3_unsure.no_delete2 == 1]
oof3_unsure_2 = oof3_unsure_2.loc[oof3_unsure_2.no_delete == 0]
# oof3_unsure_nextstep = oof3_unsure.loc[(oof3_unsure.no_delete != 1)&(oof3_unsure.no_delete2 == 1)]
oof3_unsure_1 = oof3_unsure_1.loc[oof3_unsure_1.index0 < oof3_unsure_1.index1]
oof3_unsure_2 = oof3_unsure_2.loc[oof3_unsure_2.index0 < oof3_unsure_2.index1]
oof3_unsure_1['minus'] = oof3_unsure_1.apply(lambda x : x[1][2]-x[1][3] ,axis = 1)
# best_score1 = best_score.drop('工单编号',axis = 1)
best_score_value = best_score1.values
for i in oof3_unsure_1.index:
    best_score_value[i] = result1[i]
    best_score_value[i,oof3_unsure_1.loc[i,'index0']] = best_score_value[i,oof3_unsure_1.loc[i,'index0']]*3

predictions = pd.DataFrame(best_score_value)
test_result = np.around(predictions.values,decimals = 3)
label = list(column.keys())
pd_test_result=pd.DataFrame(test_result,columns = label)
test_df = pd.read_csv(open(raw+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
del test_df['故障发生时间'],test_df['涉及告警基站或小区名称'],test_df['故障原因定位（大类）']
for i in s:
    test_df[i] = pd_test_result[i]
    test_df[i] = test_df[i].apply(lambda x: str(int(x)) if x == 0 else x)
test_df.to_csv(out_path+'result_update.csv',index = False,encoding='GB2312')


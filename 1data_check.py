# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 1data_check.py
# @time  : 2019/7/19
"""
文件说明：数据探查
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


path = "F:/项目相关/1907cm_station/data/"

train1 = pd.read_csv(open(path+"训练故障工单.csv"))
train1.columns = ["id","time","station","error"]
train2 = pd.read_csv(open(path+"测试告警.csv"))
train2.columns = ["time_alert","alert","station"]
train1["date"] = train1["time"].apply(lambda x:x[:8])
train1["station_10"] = train1["station"].apply(lambda x:int(x,16))

error = train1.groupby(["station","date"],as_index=False)["error"].agg(["count","nunique"])
error = error.reset_index()#同一天同一时刻可能由两个不同的故障
# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : nfm.py
# @time  : 2019/9/4
"""
文件说明：nfm对告警信息处理
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
# train1.columns = ["id","station","time","error"]#60346,5
# error_type={}
# n=0
# for i in train1.error.value_counts().index:
#     error_type[i]=n
#     n=n+1
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
# train1["error"] = train1["error"].replace(error_type)
train1["is_train"]=1

train2 = pd.read_csv(open(path+"训练告警.csv",encoding="gb2312"),parse_dates=["告警发生时间"])
# train2.columns = ["station","alert","time_alert"]#3401351,3 大量同一时刻2条相同数据
test1 = pd.read_csv(open(path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
# test1.columns = ["id","station","time","error"]#6696,5
test1["is_train"]=0
test2 = pd.read_csv(open(path+"测试告警.csv",encoding="gb2312"),parse_dates=["告警发生时间"])
# test2.columns = ["station","alert","time_alert"]#1212175,3
#
# error = pd.concat([train1,test1],axis=0)
# error = error.reset_index(drop=True)
# alert = pd.concat([train2,test2],axis=0)
# alert = alert.reset_index(drop=True)
# alert = alert.drop_duplicates()
#
# le = preprocessing.LabelEncoder()
# alert["alert"] = le.fit_transform(alert["alert"])
# alert_type = {index: label for index, label in enumerate(le.classes_)}

df = pd.concat([train2,test2],axis=0)
df = df.drop_duplicates()
new_data = df.groupby('涉及告警基站或小区名称')['告警标题'].apply(lambda x :'|'.join(x)).reset_index()
new_data['总标题个数'] = new_data['告警标题'].apply(lambda x: len(str(x).split('|')))
new_data['总标题_ninique'] = new_data['告警标题'].apply(lambda x: len(set(str(x).split('|'))))

data = pd.concat([train1,test1],axis=0)
data = data.merge(new_data[['涉及告警基站或小区名称', '总标题个数', '总标题_ninique']], how='left', on='涉及告警基站或小区名称')
from collections import Counter
def word_fre(x):
    word_dict = []
    x = x.split('|')
    docs = []
    for doc in x:
        doc = doc.split()
        docs.append(doc)
        word_dict.extend(doc)
    word_dict = Counter(word_dict)
    new_word_dict = {}
    for key,value in word_dict.items():
        new_word_dict[key] = [value,0]
    del word_dict
    del x
    for doc in docs:
        doc = Counter(doc)
        for word in doc.keys():
            new_word_dict[word][1] += 1
    return new_word_dict

new_data['word_fre'] = new_data['告警标题'].apply(word_fre)

def top_100(word_dict):
     return sorted(word_dict.items(),key = lambda x:(x[1][1],x[1][0]),reverse = True)[:100]
new_data['top_100'] = new_data['word_fre'].apply(top_100)

def top_100_word(word_list):
    words = []
    for i in word_list:
        i = list(i)
        words.append(i[0])
    return words


new_data['top_100_word'] = new_data['top_100'].apply(top_100_word)
print('top_100_word的shape')
print(new_data.shape)

word_list = []
for i in new_data['top_100_word'].values:
    word_list.extend(i)


word_list = Counter(word_list)
word_list = sorted(word_list.items(),key = lambda x:x[1],reverse = True)
user_fre = []
for i in word_list:
    i = list(i)
    user_fre.append(i[1]/new_data['涉及告警基站或小区名称'].nunique())

stop_words = []
for i,j in zip(word_list,user_fre):
    if j>0.5:
        i = list(i)
        stop_words.append(i[0])

new_data['title_feature'] = new_data['告警标题'].apply(lambda x: x.split('|'))
new_data['title_feature'] = new_data['title_feature'].apply(lambda line: [w for w in line if w not in stop_words])
new_data['title_feature'] = new_data['title_feature'].apply(lambda x: ' '.join(x))

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
tfidf = tfidf_vectorizer.fit_transform(new_data['title_feature'].values)
#使用nmf算法，提取文本的主题分布
text_nmf = NMF(n_components=20).fit_transform(tfidf)

name = ['taglist_'+str(x) for x in range(1,21)]
tag_list = pd.DataFrame(text_nmf)
tag_list.columns = name
tag_list['涉及告警基站或小区名称'] = new_data['涉及告警基站或小区名称']
column_name = ['涉及告警基站或小区名称'] + name
tag_list = tag_list[column_name]

tag_list.to_csv('jizhan_20.csv',index=False)

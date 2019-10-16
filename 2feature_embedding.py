# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 2feature_embedding.py
# @time  : 2019/8/1
"""
文件说明：近6小时告警作为时间序列，进行W2V，embedding
需设置环境变量 PYTHONHASHSEED = 2019
"""
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta
from gensim.models import Word2Vec
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

# path = "F:/项目相关/1907cm_station/data/"
# out = "F:/项目相关/1907cm_station/feature/"

path = "/home/dev/lm/cm_station/data/chusai/"
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

for i in [1,3,6,12]:
    print(i)

    error_union = error.merge(alert, how='left', on=["station"])
    error_union = error_union.loc[(error_union["time"] >= error_union["time_alert"]) & (
            error_union["time_alert"] > error_union["time_last{}".format(i)])]
    error_union["alert"] = error_union["alert"].astype(str).apply(lambda x: "a" + x)
    error_group = error_union.groupby(["id", "is_train"], as_index=False).agg(agg)
    error_group.columns = ["id", "is_train", "alert"]
    # error_group = error_group.sort_values(by=["is_train","id"],ascending=[False,True])
    # word2vec
    word_list = []
    for word in error_group["alert"]:
        word_list.append(word)
    model = Word2Vec(word_list, size=16, window=3,
                     min_count=1, workers=1,
                     iter=20, seed=2019)
    # 构造ids 所有告警集合
    ids = sorted(list(set(error_union["alert"].values)))
    print("alert种类", len(ids))
    # 构造vec ids对应的vec 维度为 ids个数*n
    word_vec = []
    for id in ids:
        word_vec.append(list(model[id]))
    word_vec = np.array(word_vec).astype(np.float32)
    print("w2v维度", len(word_vec), len(word_vec[0]))
    word_vec_df = pd.DataFrame(word_vec)
    params = tf.Variable(word_vec)

    #word count编码 若不加a则算法无法识别'0'~'9'
    cv = CountVectorizer(min_df=1, max_df=999999)
    error_group["alert_cv"] = error_group["alert"].apply(lambda x: " ".join(x))
    alert_cv = cv.fit_transform(error_group["alert_cv"]) #元素位置 及值
    alert_cv_df = pd.DataFrame(alert_cv.toarray())
    name = cv.vocabulary_
    name2 = sorted(name.items(), key=lambda x: x[1], reverse=False)
    name_id = [ kk[0] for kk in name2]
    print("w2v和cv顺序是否一致",ids==name_id)
    #获取word count矩阵的行列位置，及对应的值
    cv_row =list(alert_cv.tocoo().row.reshape(-1)) #cv的行
    cv_col = list(alert_cv.tocoo().col.reshape(-1)) #cv的列
    col_cnt = alert_cv.data.tolist()#col出现的次数
    cv_df = pd.DataFrame({"cv_row":cv_row,"cv_col":cv_col,"col_cnt":col_cnt})
    #按照出现的次数展开
    cv_row_new = []
    cv_col_new = []
    for ind in range(cv_df.shape[0]):
        cv_row_new.extend([cv_df["cv_row"][ind]]*cv_df["col_cnt"][ind])
        cv_col_new.extend([cv_df["cv_col"][ind]] * cv_df["col_cnt"][ind])
    cv_df_new = pd.DataFrame({"cv_row": cv_row_new, "cv_col": cv_col_new})
    cv_df_new["rank"] = cv_df_new.groupby("cv_row")["cv_col"].rank(ascending=True, method="first")
    cv_df_new["rank"] = (cv_df_new["rank"]-1).astype(int)
    #组合indices和values
    indices = []
    values = []
    for ind in range(cv_df_new.shape[0]):
        indices.append([cv_df_new["cv_row"][ind],cv_df_new["rank"][ind]])
        values.append(cv_df_new["cv_col"][ind])
    #构造embedding输入
    tags = tf.SparseTensor(indices=indices, values=values, dense_shape=(cv_df_new["cv_row"].max()+1, cv_df_new["rank"].max()+1))
    # a = tf.SparseTensor(indices=[[0, 0], [1, 2], [1, 3]], values=[1, 2, 3], dense_shape=[2, 4])
    embedding_tags = tf.nn.embedding_lookup_sparse(params, sp_ids=tags, sp_weights=None)

    with tf.Session() as s:
        s.run([tf.global_variables_initializer(), tf.tables_initializer()])
        station_alert_embedding = s.run([embedding_tags])

    station_alert_embedding = station_alert_embedding[0]
    station_alert_embedding = pd.DataFrame(station_alert_embedding, columns=['station_alert{}_em0'.format(i), 'station_alert{}_em1'.format(i),
                                                               'station_alert{}_em2'.format(i), 'station_alert{}_em3'.format(i),
                                                               'station_alert{}_em4'.format(i), 'station_alert{}_em5'.format(i),
                                                               'station_alert{}_em6'.format(i), 'station_alert{}_em7'.format(i),
                                                                             # 'station_alert{}_em8'.format(i),
                                                                             # 'station_alert{}_em9'.format(i),
                                                                             # 'station_alert{}_em10'.format(i),
                                                                             # 'station_alert{}_em11'.format(i),
                                                                             # 'station_alert{}_em12'.format(i),
                                                                             # 'station_alert{}_em13'.format(i),
                                                                             # 'station_alert{}_em14'.format(i),
                                                                             # 'station_alert{}_em15'.format(i),
                                                                             ])
    error_tmp = pd.concat([error_group[["id","is_train"]], station_alert_embedding], axis=1)
    error = error.merge(error_tmp,how = 'left',on=["id","is_train"])

del error["time"],error["station"],error["error"]
del error["time_last1"],error["time_last3"],error["time_last6"],error["time_last12"]

error.to_csv(out+"data_embedding16_0804.csv",index=False)
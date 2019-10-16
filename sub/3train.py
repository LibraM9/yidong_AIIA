# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:44:33 2019
@author: Gupeng
"""

import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
import os
raw = '/home/dev/lm/cm_station/data/fusai/' #原始文件位置
ori_path = './'
path = ori_path + 'feature/'
out_path = ori_path + "out/"
data_all = pd.read_csv(open(path+'datafusai_all0905_nodrop.csv',encoding="utf8"))
data_label = pd.read_csv(open(path+'data_label0905.csv',encoding="utf8"))
data_gupeng = pd.read_csv(open(path+'gupeng0820.csv',encoding="utf8"))
data_embedding_alert = pd.read_csv(open(path+'alert_embedding.csv',encoding="utf8"))
data_embedding_alert_all = pd.read_csv(open(path+'alert_embedding_all.csv',encoding="utf8"))
data_embedding_error = pd.read_csv(open(path+'error_embedding.csv',encoding="utf8"))
data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
data_all = data_all.merge(data_gupeng,how='left',on=["id","station","is_train"])
data_all = data_all.merge(data_embedding_error,how='left',on=['id',"station",'is_train'])
data_all = data_all.merge(data_embedding_alert,how='left',on=['id',"station",'is_train'])
data_all = data_all.merge(data_embedding_alert_all,how='left',on=['id',"station",'is_train'])
del data_all['time'],data_all['station']
train = data_all.loc[data_all.is_train == 1]
train = train.reset_index(drop=True)
###对train筛选，去除概率低的
val = True #使用验证集
n = 23 #分类数量
train_1 = train.loc[train["error"].isin(range(n))]
#####
test = data_all.loc[~(data_all['is_train'] == 1)]
test = test.reset_index(drop=True)
y_train = train_1['error']
y_test = test['error']
del train_1['error'] ,test['error'],train_1['is_train'] ,test['is_train'],train_1["id"],test["id"]
features = ["hour","day","day_of_week"]
listtag = ["time_all_alert",'time_futureall_alert',"time_last0_alert","time_last5_alert",
           "time_future10_alert","time_last10_alert","time_future5_alert","time_all_error",
           "time_futureall_error","time_last0_error",'link_last','link_future']
for tag in listtag:
    features = features+[ i for i in train.columns if tag in i]
###########################开始训练
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error,roc_auc_score,accuracy_score
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np
# listfold = [100,2333,9000,100000,19960115]
listfold = [2333]
for foldnum in listfold:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=foldnum)
    # folds = KFold(n_splits=5, shuffle=True, random_state=2333)
    oof = np.zeros((len(train_1),n))
    oof_train = np.zeros((len(train),n))
    predictions = np.zeros((len(test),n))
    feature_importance_df = pd.DataFrame()
    train_x = train_1[features]
    test_x = test[features]
    clf_labels = y_train
    param = {'objective': 'multiclass',
             'num_class':n,
             'num_leaves': 2**6, #2**5
             # 'min_data_in_leaf': 25,#
             'max_depth': 6,  # 5 2.02949 4 2.02981
             'learning_rate': 0.02, #0.02
             'lambda_l1': 0.13,
             "boosting": "gbdt",
             "feature_fraction": 0.85,
             'bagging_freq': 8,
             "bagging_fraction": 0.9, #0.9
             "metric": 'multi_logloss',
             "verbosity": -1,
             "random_state": 2333,
             "num_threads" : 50}
    categorical_feats = []
    # lgb
    model = "lgb"
    temp = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, clf_labels)):
        print("fold {}".format(fold_))
        trn_data = lgb.Dataset(train_x.iloc[trn_idx][features],label=clf_labels.iloc[trn_idx],categorical_feature=categorical_feats)
        val_data = lgb.Dataset(train_x.iloc[val_idx][features],label=clf_labels.iloc[val_idx],categorical_feature=categorical_feats)
        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=50,early_stopping_rounds=100)
        #n*6矩阵
        oof[val_idx] = clf.predict(train_x.iloc[val_idx][features], num_iteration=clf.best_iteration)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof_train +=clf.predict(train_x.iloc[:][features], num_iteration=clf.best_iteration) / folds.n_splits
        predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits
    print("CV score: {:<8.5f}".format(log_loss(clf_labels, oof)))
    pd.DataFrame(oof).to_csv(out_path+"oof_{}.csv".format(model),index=False)
    oof_train = pd.DataFrame(oof_train)
    if n < 6:
        for i in range(n,6):
            oof_train[i]=0
    print("CV score_train: {:<8.5f}".format(log_loss(train["error"].values, oof_train.values)))
    feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
    ascending=False)
    #评价指标： AUC指标 准确率
    if val==False:
        oof = oof_train.values
    clf_one_hot = train["error"]
    clf_one_hot = pd.get_dummies(clf_one_hot)
    auc = roc_auc_score(clf_one_hot, oof,average='weighted')
    print("auc:",auc)
    oof_max = np.argmax(oof,axis=1)
    #1. 找出value前n大的数所对应的的列：
    def accuary_multscore(label,value,n):
        topn = np.array([[] for i in range(len(label))])
        for i in range(n):
            arg1 = np.argmax(value, axis=1)
            topn = np.hstack((topn, arg1.reshape(len(arg1),1)))
            for j in range(len(value)):
                value[j][arg1[j]] = -1
        list1 = [1 if (label[i] in topn[i]) else 0 for i in range(len(label)) ]
        return sum(list1)/len(label)

    import heapq
    def acc_new(y_true,oof):
        oof2 = pd.Series(list(oof))
        oof2=oof2.apply(lambda x:list(map(list(x).index, heapq.nlargest(3,x))))
        ans=[]
        for i in range(len(oof2)):
            if y_true[i] in oof2[i]:
                ans.append(1)
            else:
                ans.append(0)
        return sum(ans)/len(ans)
    acc = acc_new(train["error"].values,oof)
    # acc = accuracy_score(train["error"].values,oof_max)
    # acc = accuary_multscore(train["error"].values,oof_max,3)
    print("acc:",acc)
    print("result:",0.5*auc+0.5*acc)
    loss = str(0.5*auc+0.5*acc)[2:6]
    predictions = pd.DataFrame(predictions)

    if n < 6:
        for i in range(n,6):
            predictions[n]=0
    test_result = np.around(predictions.values,decimals = 3)

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
    label = list(column.keys())
    pd_test_result=pd.DataFrame(test_result,columns = label)
    test_df = pd.read_csv(open(raw+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
    del test_df['故障发生时间'],test_df['涉及告警基站或小区名称'],test_df['故障原因定位（大类）']
    for i in s:
        test_df[i] = pd_test_result[i]
        test_df[i] = test_df[i].apply(lambda x: str(int(x)) if x == 0 else x)
    exec('''folf_result_{} = test_df'''.format(foldnum))

test_df.to_csv(out_path+'result.csv',index = False,encoding='GB2312')

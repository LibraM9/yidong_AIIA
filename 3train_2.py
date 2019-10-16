# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3train_2.py
# @time  : 2019/7/29
"""
文件说明：二分类模型训练
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

ori_path = "/home/dev/lm/cm_station/data/"
# path = "F:/项目相关/1907cm_station/feature/"
path = "/home/dev/lm/cm_station/feature/"
out_path = "/home/dev/lm/cm_station/out/"

data_all = pd.read_csv(open(path+'data_all0725.csv',encoding="utf8")) #线下0.8745 线上
data_label = pd.read_csv(open(path+'data_label0725.csv',encoding="utf8")) #线下0.8745 线上
data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
#构建二分类label
for i in range(6):
    data_all["is_{}".format(i)] = data_all["error"].apply(lambda x: 1 if x==i else 0)

train = data_all.loc[data_all.is_train == 1]
train = train.reset_index(drop=True)

#####
test = data_all.loc[~(data_all['is_train'] == 1)]
test = test.reset_index(drop=True)

features = ["hour","day","day_of_week"]
features = features+[ i for i in train.columns if "time_all_alert" in i]\
           +[ i for i in train.columns if "time_futureall_alert" in i]+[ i for i in train.columns if "time_last0_alert" in i]\
           +[ i for i in train.columns if "time_last5_alert" in i]+[ i for i in train.columns if "time_future5_alert" in i]\
           +[ i for i in train.columns if "time_all_error" in i]+[ i for i in train.columns if "time_futureall_error" in i]\
           +[ i for i in train.columns if "time_last0_error" in i]
# features = train.columns.values
"""
all 0.8417
futureall 0.8309
0 0.8187
all+futureall+0 0.8719
all+futureall+0+5 0.8764
all+futureall+last0+last5+future5 0.8768
all+futureall+last0+last5+future5+allerror 0.8927 /0.8941
all+futureall+last0+last5+future5+allerror+futureallerror 0.8949
all+futureall+last0+last5+future5+allerror+futureallerror+last0error lr 0.02 lgb 0.8977/0.8976

"""

###########################开始训练
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error,roc_auc_score,accuracy_score
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier

auc = []
for i in range(6):
    print(i)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    train_x = train[features].values
    test_x = test[features].values

    y = 'is_{}'.format(i)
    y_train = train[y]
    y_test = test[y]
    clf_labels = y_train.values

    if i in [0,1,2,3]:
        param = {'objective': 'binary',
                 'num_leaves': 2**6, #2**5
                 # 'min_data_in_leaf': 10,#
                 'max_depth': 6,  # 5 2.02949 4 2.02981
                 'learning_rate': 0.02, #0.02
                 'lambda_l1': 0.13,
                 "boosting": "gbdt",
                 "feature_fraction": 0.7,#0.85
                 'bagging_freq': 8,
                 "bagging_fraction": 0.9, #0.9
                 "metric": 'auc',
                 'is_unbalance': True,
                 "verbosity": -1,
                 "random_state": 2333,
                 "num_threads" : 50}
    else:
        param = {'objective': 'binary',
                 'num_leaves': 2 ** 6,  # 2**5
                 'min_data_in_leaf': 25,#
                 'max_depth': 6,  # 5 2.02949 4 2.02981
                 'learning_rate': 0.02,  # 0.02
                 'lambda_l1': 0.13,
                 "boosting": "gbdt",
                 "feature_fraction": 0.7,#0.85
                 'bagging_freq': 8,
                 "bagging_fraction": 0.8,  # 0.9
                 "metric": 'auc',
                 'is_unbalance': True,
                 "verbosity": -1,
                 "random_state": 2333,
                 "num_threads": 50}
    # lgb
    model = "lgb"
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, clf_labels)):
        print("fold {}".format(fold_))
        trn_data = lgb.Dataset(train_x[trn_idx],label=clf_labels[trn_idx])
        val_data = lgb.Dataset(train_x[val_idx],label=clf_labels[val_idx])

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,early_stopping_rounds=100)
        #n*6矩阵
        oof[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

    print("auc score:", roc_auc_score(train[y].values, oof))
    feature_importance = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",ascending=False)

    train_dic = {
        "id":train["id"],
        "is_train":train["is_train"],
        "is_{}".format(i):oof
    }
    train_prob = pd.DataFrame(train_dic)
    train_prob.to_csv(out_path+"train_is_{}.csv".format(i),index=None)
    test_dic = {
        "id":test["id"],
        "is_train":test["is_train"],
        "is_{}".format(i):predictions
    }
    test_prob = pd.DataFrame(test_dic)
    test_prob.to_csv(out_path+"test_is_{}.csv".format(i),index=None)

    auc.append(roc_auc_score(train[y].values, oof))

print(auc)


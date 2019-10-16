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

ori_path = "/home/dev/lm/cm_station/data/"
# path = "F:/项目相关/1907cm_station/feature/"
path = "/home/dev/lm/cm_station/feature/"
out_path = "/home/dev/lm/cm_station/out/"

data_all = pd.read_csv(path + 'data_all_gp.csv') #线下0.8573 线上0.8620
del data_all['time'], data_all['station']

train = data_all.loc[data_all.is_train == 1]
test = data_all.loc[~(data_all['is_train'] == 1)]
y_train = train['error']
y_test = test['error']
del train['error'], test['error'], train['is_train'], test['is_train'], train['id'], test['id']

features = ["hour","day","day_of_week"]
features = features+[ i for i in train.columns if "time_last0" in i]+[ i for i in train.columns if "time_last5" in i]
# features = train.columns.values

train_values = train[features].values
test_values = test[features].values
clf_labels = y_train.values



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
clf = LGBMClassifier(
    learning_rate=0.05,
    n_estimators=10000,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=2019,
    max_depth=6
)

oof = np.zeros((len(train),6))
predictions = np.zeros((len(test),6))
feature_importance_df = pd.DataFrame()

for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels)):
    print(i, 'fold...')
    trn_x, trn_y = train_values[trn_idx], clf_labels[trn_idx]
    val_x, val_y = train_values[val_idx], clf_labels[val_idx]

    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        early_stopping_rounds=100, verbose=50
    )

    oof[val_idx] = clf.predict_proba(train_values[val_idx], num_iteration=clf.best_iteration_)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] =  + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict_proba(test_values, num_iteration=clf.best_iteration_) / skf.n_splits

# 评价指标： AUC指标 准确率
clf_one_hot = pd.Series(clf_labels)
clf_one_hot = pd.get_dummies(clf_one_hot)
auc = roc_auc_score(clf_one_hot, oof,average='weighted')
print("auc:",auc)
oof_max = np.argmax(oof,axis=1)
acc = accuracy_score(clf_labels,oof_max)
print("acc:",acc)
print("result:",0.5*auc+0.5*acc)

feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
ascending=False)

test_result = np.around(predictions, decimals=3)
pd_test_result = pd.DataFrame(test_result, columns=['传输故障', '动环故障', '电力故障', '硬件故障', '误报警', '软件故障'])

test_df = pd.read_csv(open(ori_path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
del test_df['故障发生时间'],test_df['涉及基站eNBID或小区ECGI'],test_df['故障原因定位（大类）']

for i in ['电力故障', '传输故障', '软件故障', '硬件故障', '动环故障', '误报警']:
    test_df[i] = pd_test_result[i]
    test_df[i] = test_df[i].apply(lambda x: str(int(x)) if x == 0 else x)

test_df.to_csv(out_path+'result0722.csv',index = False,encoding='GB2312')

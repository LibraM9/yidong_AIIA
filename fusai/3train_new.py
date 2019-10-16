# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3train_new.py
# @time  : 2019/9/6
"""
文件说明：
"""
import pandas as pd
import numpy as np
# np.random.seed(2333)
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn import preprocessing

import sys
sys.path.append("/home/dev/lm/utils_lm")

ori_path = "/home/dev/lm/cm_station/data/fusai/"
path = "/home/dev/lm/cm_station/fusai/"
out_path = "/home/dev/lm/cm_station/out/fusai/"

data_all = pd.read_csv(open(path+'datafusai_all0905_nodrop.csv',encoding="utf8"))
data_label = pd.read_csv(open(path+'data_label0905.csv',encoding="utf8"))
data_embedding = pd.read_csv(open(path+'alert_embedding_new.csv',encoding="utf8"))
data_nfm = pd.read_csv(open(path+'jizhan_nini_20.csv',encoding="utf8"))

data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
data_all = data_all.merge(data_embedding,how='left',on="station")
data_all = data_all.merge(data_nfm,how='left',left_on='station',right_on="涉及告警基站或小区名称")

train = data_all.loc[data_all.is_train == 1]
train = train.reset_index(drop=True)
###对train筛选，去除概率低的

n = 23 #分类数量
if n<23:
    val = False  # 使用验证集
else:
    val = True
train_1 = train.loc[train["error"].isin(range(n))]
#####
test = data_all.loc[~(data_all['is_train'] == 1)]
test = test.reset_index(drop=True)
y_train = train_1['error']
y_test = test['error']

features = ["hour","day","day_of_week"]
features = features\
+[ i for i in train.columns if "taglist_" in i]\
+['0','1','2','3']\
+[ i for i in train.columns if "time_all_error_" in i]\
+[ i for i in train.columns if "time_last5_error_" in i]\
+[ i for i in train.columns if "time_last60_error_" in i]\
+[ i for i in train.columns if "time_all_alert_" in i]\
+[ i for i in train.columns if "time_last5_alert_" in i]\
+[ i for i in train.columns if "time_lastall_alert_" in i]\
+[ i for i in train.columns if "time_future5_alert_" in i]\

print(len(features))
"""
nfm 0.8710
nfm+emb 0.8797
nfm+emb+errorall 0.9191
nfm+emb+errorall+last60 0.9194
nfm+emb+errorall+last60_last5 0.9197

nfm+emb+errorall+last60_last5+alertall 0.9209
nfm+emb+errorall+errorlast60_errorlast5+alertall_alertlast5 0.9216
nfm+emb+errorall+errorlast60_errorlast5+alertall_alertlast5+alertlastall 0.9219
nfm+emb+errorall+errorlast60_errorlast5+alertall_alertlast5+alertlastall+alertfuture5 0.9312

"""
###########################开始训练
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error,roc_auc_score,accuracy_score
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
# folds = KFold(n_splits=5, shuffle=True, random_state=2333)

oof = np.zeros((len(train_1),n))
oof_train = np.zeros((len(train),n))
predictions = np.zeros((len(test),n))
feature_importance_df = pd.DataFrame()

train_x = train_1[features].values
test_x = test[features].values
clf_labels = y_train.values

#新的acc函数
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
# oht = preprocessing.OneHotEncoder(sparse = False)
# oht_test=oht.fit_transform(train["error"].values.reshape(len(train),1))
# def custom_mertic(preds,train_data):
#     '''
#     input:
#         train_data: np.array of size [n_sample,]
#         preds: np.array of size [n_sample * n_class, ]
#     return:
#         return (eval_name, eval_result, is_higher_better)
#     is_higher_better : bool
#     是否越高越好, e.g. AUC is ``is_higher_better``.
#     '''
#     y_true = train_data.get_label()
#     y_true_oht=oht.transform(np.reshape(y_true,[len(y_true),1]))
#     y_pred = np.reshape(preds, [len(y_true), n], order='F')
#     auc = roc_auc_score(y_true_oht, y_pred, average='weighted')
#     acc = acc_new(y_true,y_pred)
#     return "custom_mertic",0.5*auc+0.5*acc,True

param = {'objective': 'multiclass',
         'num_class':n,
         'num_leaves': 2**6, #2**5
         # 'min_data_in_leaf': 25,#
         'max_depth': 6,  # 6
         'learning_rate': 0.02, #0.02
         'lambda_l1': 0.13,
         "boosting": "gbdt",
         "feature_fraction": 0.85,#0.85
         'bagging_freq': 8,
         "bagging_fraction": 0.9, #0.9
         # "metric": 'multi_logloss',
         "verbosity": -1,
         "random_state": 2333,
         "num_threads" : 50}
# # lgb
model = "lgb"
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, clf_labels)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x[trn_idx],label=clf_labels[trn_idx])
    val_data = lgb.Dataset(train_x[val_idx],label=clf_labels[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data]
                    # , feval=custom_mertic
                    , verbose_eval=200,early_stopping_rounds=100)
    #n*6矩阵
    oof[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    oof_train +=clf.predict(train[features].values, num_iteration=clf.best_iteration) / folds.n_splits
    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(clf_labels, oof)))
pd.DataFrame(oof).to_csv(out_path+"oof_{}.csv".format(model),index=False)
oof_train = pd.DataFrame(oof_train)
if n < 23:
    for i in range(n,23):
        oof_train[i]=0
print("CV score_train: {:<8.5f}".format(log_loss(train["error"].values, oof_train.values)))
feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
ascending=False)
#评价指标： AUC指标 准确率
if val==False:
    oof = oof_train.values #6 0.9342 5 0.9328/0.8969
clf_one_hot = train["error"]
clf_one_hot = pd.get_dummies(clf_one_hot)
auc = roc_auc_score(clf_one_hot, oof,average='weighted')
print("auc:",auc)
#计算acc
acc = acc_new(train["error"].values,oof)
print("acc:",acc)
print("result:",0.5*auc+0.5*acc)
loss = str(0.5*auc+0.5*acc)[2:6]

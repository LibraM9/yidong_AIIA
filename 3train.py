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
from sklearn import preprocessing

import os

ori_path = "/home/dev/lm/cm_station/data/fusai/"
# path = "F:/项目相关/1907cm_station/feature/"
path = "/home/dev/lm/cm_station/feature/fusai/"
out_path = "/home/dev/lm/cm_station/out/fusai/"

data_all = pd.read_csv(open(path+'data_all0809.csv',encoding="utf8")) #线下0.8745 线上
data_label = pd.read_csv(open(path+'data_label0809.csv',encoding="utf8")) #线下0.8745 线上
# data_gp = pd.read_csv(open(path+'gupengfeaturenew.csv',encoding="utf8")) #线下0.8745 线上
data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
# data_all = data_all.merge(data_gp,how='left',on=["id","station","is_train"])
del data_all['time'],data_all['station']

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

del train_1['error'] ,test['error'],train_1['is_train'] ,test['is_train'],train_1["id"],test["id"]

features = ["hour","day","day_of_week"]
features = features+[ i for i in train.columns if "time_all_alert" in i]\
           +[ i for i in train.columns if "time_futureall_alert" in i] \
           +[ i for i in train.columns if "time_future2_alert" in i]\
           # +[ i for i in train.columns if "time_future5_alert" in i]\
           # +[ i for i in train.columns if "time_all_error" in i]+[ i for i in train.columns if "time_futureall_error" in i]\
           # +[ i for i in train.columns if "time_last0_error" in i]
            # + [i for i in train.columns if "time_last0_alert" in i] \
            # + [i for i in train.columns if "time_last2_alert" in i] \

    # features = train.columns.values
"""
all 0.8828
all+futureall 0.8866
all+futureall+0 0.8861
all+futureall+0+last2 0.8864
all+futureall+0+last2+future2 0.8962
all+futureall+future2 0.8968
all+futureall+future2+last5 0.8966

all+futureall+last2 0.8863
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
oht = preprocessing.OneHotEncoder()
oht_test=oht.fit_transform(train["error"])
def custom_mertic(y_true,y_pred):
    y_pred = np.reshape(y_pred, [len(train), 3], order='F')
    return "custom_mertic"
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
# lgb
model = "lgb"
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, clf_labels)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x[trn_idx],label=clf_labels[trn_idx])
    val_data = lgb.Dataset(train_x[val_idx],label=clf_labels[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=50,early_stopping_rounds=100)
    #n*6矩阵
    oof[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    oof_train +=clf.predict(train[features].values, num_iteration=clf.best_iteration) / folds.n_splits
    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

#catboost
# model = "cat"
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, clf_labels)):
#     print("fold {}".format(fold_))
#     trn_x, trn_y = train_x[trn_idx], clf_labels[trn_idx]
#     val_x, val_y = train_x[val_idx], clf_labels[val_idx]
#     num_round = 10000
#     clf = CatBoostClassifier(
#         objective="MultiClass",# MultiClass 0.8957
#         custom_metric="MultiClass",
#         eval_metric="MultiClass",
#         n_estimators=num_round,
#         depth=6,
#         learning_rate=0.08,
#         num_leaves=31,
#         l2_leaf_reg=0.5, #0.5 0.8966
#         min_data_in_leaf=1,#
#         random_seed=2019,
#         # class_weights=[1,3,3,6,6,6],
#         thread_count=-1,
#         verbose=True)
#     clf.fit(trn_x,trn_y,
#             eval_set=[(trn_x, trn_y), (val_x, val_y)],
#             early_stopping_rounds=100, verbose=50
#             # cat_features=cat_features
#             )
#     #n*6矩阵
#     oof[val_idx] = clf.predict_proba(train_x[val_idx])
#
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["Feature"] = features
#     fold_importance_df["importance"] = clf.feature_importances_
#     fold_importance_df["fold"] = fold_ + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
#
#     oof_train +=clf.predict_proba(train[features].values) / folds.n_splits
#     predictions += clf.predict_proba(test_x) / folds.n_splits

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

predictions = pd.DataFrame(predictions)
if n < 6:
    for i in range(n,6):
        predictions[n]=0
test_result = np.around(predictions.values,decimals = 3)
pd_test_result=pd.DataFrame(test_result,columns = ["电力故障","硬件故障","传输故障","软件故障","动环故障","误告警"])

test_df = pd.read_csv(open(ori_path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
del test_df['故障发生时间'],test_df['涉及基站eNBID或小区ECGI'],test_df['故障原因定位（大类）']

for i in ['电力故障', '传输故障', '软件故障', '硬件故障', '动环故障', '误告警']:
    test_df[i] = pd_test_result[i]
    test_df[i] = test_df[i].apply(lambda x: str(int(x)) if x == 0 else x)

test_df.to_csv(out_path+'result_{}{}_{}.csv'.format(model,n,loss),index = False,encoding='GB2312')

###########


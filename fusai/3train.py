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

import sys
sys.path.append("/home/dev/lm/utils_lm")

ori_path = "/home/dev/lm/cm_station/data/fusai/"
# path = "F:/项目相关/1907cm_station/feature/"
path = "/home/dev/lm/cm_station/feature/fusai/"
path = "/home/dev/lm/cm_station/fusai/"
out_path = "/home/dev/lm/cm_station/out/fusai/"

# data_all = pd.read_csv(open(path+'data_all0809.csv',encoding="utf8")) #
# data_label = pd.read_csv(open(path+'data_label0809.csv',encoding="utf8")) #
# data_gp = pd.read_csv(open(path+'gupeng0814.csv',encoding="utf8")) #
# data_embedding = pd.read_csv(open(path+'data_embedding16_0818.csv',encoding="utf8")) #
# data_gp = pd.read_csv(open(path+'gupengfeaturenew.csv',encoding="utf8")) #
# data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
# data_all = data_all.merge(data_gp,how='left',on=["id","station","is_train"])
# data_all = data_all.merge(data_embedding,how='left',on=["id","is_train"])
# data_all = data_all.merge(data_gp,how='left',on=["id","station","is_train"])

data_all = pd.read_csv(open(path+'datafusai_all0905_nodrop.csv',encoding="utf8"))
data_label = pd.read_csv(open(path+'data_label0905.csv',encoding="utf8"))
data_embedding = pd.read_csv(open(path+'alert_embedding_new.csv',encoding="utf8"))
data_nfm = pd.read_csv(open(path+'jizhan_nini_20.csv',encoding="utf8"))

data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
data_all = data_all.merge(data_embedding,how='left',on="station")
data_all = data_all.merge(data_nfm,how='left',left_on='station',right_on="涉及告警基站或小区名称")
#########同值率筛选########

# from model_train.a2_feature_selection import select_primaryvalue_ratio
# features = select_primaryvalue_ratio(data_all,ratiolimit=0.95)[0]
# features = list(data_all.columns)
# features.remove("id")
# features.remove("station")
# features.remove("time")
# features.remove("error")
# features.remove("is_train")
####增加降维特征##################

# from sklearn.decomposition import PCA,NMF
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import FactorAnalysis,FastICA
# from sklearn.preprocessing import MinMaxScaler,StandardScaler
#
# train_test = data_all[features]
# for i in train_test.columns:
#     train_test[i] = train_test[i].fillna(train_test[i].median())
# train_test_feature = MinMaxScaler().fit_transform(train_test)
#
# pca = PCA(n_components=100)
# nmf = NMF(n_components= 50)
# fca = FactorAnalysis(n_components=70)
# ica = FastICA(n_components=70)
#
# pca.fit(train_test_feature)
# nmf.fit(train_test_feature * (train_test_feature>0))
# fca.fit(train_test_feature)
# ica.fit(train_test_feature)
#
# nmf_feature = nmf.transform(train_test_feature* (train_test_feature>0))
# nmf_feature = pd.DataFrame(nmf_feature)
# nmf_feature.columns = ["nfm_"+str(i) for i in nmf_feature.columns]
# fca_feature = fca.transform(train_test_feature)
# fca_feature = pd.DataFrame(fca_feature)
# fca_feature.columns = ["fca_"+str(i) for i in fca_feature.columns]
# ica_feature = ica.transform(train_test_feature)
# ica_feature = pd.DataFrame(ica_feature)
# ica_feature.columns = ["ica_"+str(i) for i in ica_feature.columns]
# pca_feature = pca.transform(train_test_feature)
# pca_feature = pd.DataFrame(pca_feature)
# pca_feature.columns = ["pca_"+str(i) for i in pca_feature.columns]
#
# data_all = pd.concat([data_all,nmf_feature,fca_feature,ica_feature,pca_feature],axis=1)

###########################
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
+[ i for i in train.columns if "taglist_" in i]
           # +[ i for i in train.columns if "time_all_alert" in i]\
           # +[ i for i in train.columns if "time_futureall_alert" in i]\

           # +[ i for i in train.columns if "time_last5_alert" in i]\
           # +[ i for i in train.columns if "time_future10_alert" in i]\
# +[ i for i in train.columns if "time_last10_alert" in i]\
           # + [i for i in train.columns if "time_future5_alert" in i] \
           # +[ i for i in train.columns if "time_all_error" in i]\
           # +[ i for i in train.columns if "time_futureall_error" in i]\
           # +[ i for i in train.columns if "time_last0_error" in i]\
           # +[i for i in train.columns if "link_" in i] # last 0.9316/0.9336

#lm 最佳
# features = features + [i for i in train.columns if "time_all_alert" in i] \
#            + [i for i in train.columns if "time_futureall_alert" in i]  \
#             + [i for i in train.columns if "time_future2_alert" in i] \
#            + [i for i in train.columns if "time_all_error" in i] \
#            + [i for i in train.columns if "time_futureall_error" in i] \
#            + [i for i in train.columns if "time_last0_error" in i]\

            # +[i for i in train.columns if  "pca" in i]\
            # +[i for i in train.columns if  "nfm" in i]\
            # +[i for i in train.columns if  "fca" in i]\
            # +[i for i in train.columns if  "ica" in i]
"""
all 0.8828
all+futureall 0.8866
all+futureall+0 0.8861
all+futureall+0+last2 0.8864
all+futureall+0+last2+future2 0.8962
all+futureall+future2 0.8968
all+futureall+future2+last5 0.8966
all+futureall+future2+future5 0.8966
all+futureall+last0+last5+future5+allerror+futureallerror+last0error lgb 0.9320/0.9324
all+futureall+future5+allerror+futureallerror+last0error lgb 0.9321
all+futureall+future2+allerror+futureallerror+last0error+future5error lgb 0.9321
all+futureall+future2+allerror+futureallerror+last0error lgb (0.9430 0.9219)0.9324
all+futureall+future2+allerror+futureallerror+last0error lgb 自定义函数 (0.9428 0.9223)0.9326/0.9311

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
oht = preprocessing.OneHotEncoder(sparse = False)
oht_test=oht.fit_transform(train["error"].values.reshape(len(train),1))
def custom_mertic(preds,train_data):
    '''
    input:
        train_data: np.array of size [n_sample,]
        preds: np.array of size [n_sample * n_class, ]
    return:
        return (eval_name, eval_result, is_higher_better)
    is_higher_better : bool
    是否越高越好, e.g. AUC is ``is_higher_better``.
    '''
    y_true = train_data.get_label()
    y_true_oht=oht.transform(np.reshape(y_true,[len(y_true),1]))
    y_pred = np.reshape(preds, [len(y_true), n], order='F')
    auc = roc_auc_score(y_true_oht, y_pred, average='weighted')
    acc = acc_new(y_true,y_pred)
    return "custom_mertic",0.5*auc+0.5*acc,True

param = {'objective': 'multiclass',
         'num_class':n,
         'num_leaves': 2**6, #2**5
         # 'min_data_in_leaf': 25,#
         'max_depth': 2,  # 6
         'learning_rate': 0.1, #0.02
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

# #catboost
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
if n < 23:
    for i in range(n,23):
        predictions[n]=0

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
test_result = np.around(predictions.values,decimals = 3)
pd_test_result=pd.DataFrame(test_result,columns = label)

test_df = pd.read_csv(open(ori_path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
del test_df['故障发生时间'],test_df['涉及告警基站或小区名称'],test_df['故障原因定位（大类）']

sample = pd.read_csv(open(ori_path+"sample.csv",encoding="gb2312"))
del sample["工单号"]

for i in sample.columns:
    test_df[i] = pd_test_result[i]
    test_df[i] = test_df[i].apply(lambda x: str(int(x)) if x == 0 else x)

test_df.to_csv(out_path+'result_{}{}_{}.csv'.format(model,n,loss),index = False,encoding='GB2312')

###########

from sklearn.decomposition import NMF


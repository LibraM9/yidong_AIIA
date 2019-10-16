# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 4rule.py
# @time  : 2019/7/29
"""
文件说明：模型融合及规则
"""
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

ori_path = "/home/dev/lm/cm_station/data/"
# path = "F:/项目相关/1907cm_station/feature/"
path = "/home/dev/lm/cm_station/feature/"
out_path = "/home/dev/lm/cm_station/out/"

data_all = pd.read_csv(open(path+'data_all0725.csv',encoding="utf8")) #线下0.8745 线上
data_label = pd.read_csv(open(path+'data_label0725.csv',encoding="utf8")) #线下0.8745 线上
data_gp = pd.read_csv(open(path+'gupengfeaturenew.csv',encoding="utf8")) #线下0.8745 线上
data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
del data_all['time'],data_all['station']

train = data_all.loc[data_all.is_train == 1]
train = train.reset_index(drop=True)

# 模型融合
oof_cat = pd.read_csv(out_path+"oof_cat.csv") #0.8966
oof_lgb = pd.read_csv(out_path+"oof_lgb.csv") #0.8977/0.8976
oof_dfm = pd.read_csv(out_path+"oof_dfm.csv") #0.9018/0.8932
#lgb dfm 28 0.9030 37 0.9031 46 0.9024 64 0.9011 73 0.9005
#lgb cat 82 0.8982 73 0.8982 64 0.8979
#lgb cat dfm 0.9007/0.8974
oof = (0.6*oof_lgb+0.2*oof_cat+0.2*oof_dfm).values

#二分类覆盖
def cover(model,is_num,threshold):
    model = model.copy()
    is_num = is_num.copy()
    model["id"] = is_num["id"]
    class_ratio = is_num.columns[-1]
    c = class_ratio[-1]
    print("对{}进行覆盖".format(c))
    is_num["rank"] = is_num[class_ratio].rank(ascending=False, method='first')
    th = int(is_num.shape[0] * threshold) #阈值
    print("覆盖的样本量:",th)
    print("分类阈值:", is_num[is_num["rank"]==th][class_ratio].values)
    cover_id = set(is_num[is_num["rank"] <= th]["id"])  # 提取符合条件的id
    model.loc[model["id"].isin(cover_id),c]=1
    del model["id"]
    return model

p = [0.7755947270064145,
 0.10717713015808879,
 0.09929794434062326,
 0.011616748320622254,
 0.0056063437547350875,
 0.0007071064195161372]
is_0 = pd.read_csv(out_path+"train_is_0.csv") #0.897729
oof_df = cover(oof_lgb,is_0,p[0]*0)
is_1 = pd.read_csv(out_path+"train_is_1.csv") #
oof_df = cover(oof_df,is_1,p[1]*0)
is_2 = pd.read_csv(out_path+"train_is_2.csv") #
oof_df = cover(oof_df,is_2,p[2]*0)
is_3 = pd.read_csv(out_path+"train_is_3.csv") #
oof_df = cover(oof_df,is_3,p[3]*0)
is_4 = pd.read_csv(out_path+"train_is_4.csv") #
oof_df = cover(oof_df,is_4,p[4]*0)
is_5 = pd.read_csv(out_path+"train_is_5.csv") #
oof_df = cover(oof_df,is_5,p[5]*0)
#验证
oof = oof_df.values
clf_one_hot = train["error"]
clf_one_hot = pd.get_dummies(clf_one_hot)
auc = roc_auc_score(clf_one_hot, oof,average='weighted')
print("auc:",auc)
oof_max = np.argmax(oof,axis=1)
acc = accuracy_score(train["error"].values,oof_max)
print("acc:",acc)
print("result:",0.5*auc+0.5*acc)

sub_lgb = pd.read_csv(out_path+"result_lgb8977.csv",encoding="GB2312") #0.8977/0.8976
sub_cat = pd.read_csv(out_path+"result_cat8966.csv",encoding="GB2312") #0.8966
sub_dfm = pd.read_csv(out_path+"result_dfm9031.csv",encoding="GB2312") #0.9031
sub = 0.6*sub_lgb+0.2*sub_cat+0.2*sub_dfm
sub = pd.DataFrame(np.around(sub.values,decimals = 3))
sub.columns = sub_lgb.columns
sub["工单编号"] = sub["工单编号"].astype(int)
sub.to_csv(out_path+'stack_lcd622.csv',index = False,encoding='GB2312')
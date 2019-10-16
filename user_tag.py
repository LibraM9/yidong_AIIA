# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:49:21 2019

@author: Gupeng
"""

import pandas as pd
import tensorflow as tf
#from sklearn.feature_extraction.text import CountVectorizer
path = '/home/dev/lm/paipai/ori_data/'
'''
主要使用了Ebeeding的思想，将特征由5600维变为8维  经验证 8维是最佳长度
'''


#将特征 转变为一个查找表

def sparse_from_csv(csv):
  ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
  table = tf.contrib.lookup.index_table_from_tensor(
      mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
  split_tags = tf.string_split(post_tags_str, "|")
  return tf.SparseTensor(
      indices=split_tags.indices,
      values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
      dense_shape=split_tags.dense_shape)

def  array2string(array):
    return str(array[0])+','+array[1]


'''
特征转化为 [ 1，a|b|c|d,
            2, a|f
             3,]
这种形式
'''
user_tag_df = pd.read_csv(path+'user_taglist.csv', parse_dates=['insertdate'])
user_tag_df.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)
user_tag_df = user_tag_df.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)
user_to_tensor =user_tag_df[['user_id','taglist']]
user_values = user_to_tensor.values
tag_list =  user_to_tensor.taglist.values


user_temp = []
for i in user_values:
    user_temp.append(array2string(i)) # 847942,271|5639|1314|404|2017
TAG_SET = {}
for i in tag_list:
    for tag in i.split('|'):
        TAG_SET[tag] = 1
TAG_SET = list(TAG_SET.keys())  # 所有tag的集合

#输出的维数
TAG_EMBEDDING_DIM = 8
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))
#tf.truncated_normal(shape, mean, stddev) :这个函数产生正态分布，均值和标准差自己设定
# tf.Variable是一个Variable类通过变量维持图graph的状态,以便在sess.run()中执行

#sp_weights=None代表的每一个取值的权重，如果是None的话，所有权重都是1，也就是相当于取了平均。
#如果不是None的话，我们需要同样传入一个SparseTensor，代表不同球员的喜欢权重。
tags = sparse_from_csv(user_temp)
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)
embedded_tags = tf.nn.embedding_lookup(embedding_params)
with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  user_tag_embedding = s.run([embedded_tags])

user_tag_embedding = user_tag_embedding[0]
user_tag_embedding = pd.DataFrame(user_tag_embedding,columns = ['user_tag_em0','user_tag_em1',
                                           'user_tag_em2','user_tag_em3',
                                           'user_tag_em4','user_tag_em5',
                                           'user_tag_em6','user_tag_em7',
                                           ],index = user_tag_df.index)


temp = pd.concat([user_tag_df,user_tag_embedding],axis= 1)
del temp['taglist']
temp.to_csv('my_user_taglist_embedding.csv', index=False)















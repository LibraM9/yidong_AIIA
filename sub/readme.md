2.1 feature.py : 告警统计信息

2.1feature_label: 故障统计信息

2gupeng_fusai_feature.py:故障的关联统计

embedding.py: 告警的embedding

error_embedding.py:故障的embedding

3train.py: 训练以及输出

youhua.py: 对结果进行优化


feature内的文件：
alert_embedding.csv: 告警的embedding

alert_embedding_all.csv:告警的全量embedding

error_embedding.csv:故障的embedding

data_label0905.csv: 故障的统计信息

datafusai_all0905_nodrop.csv 告警的统计信息

gupeng0820.csv: 故障的关联统计

result.csv: 输出结果

核心思路：
1.	按照时间顺序，求每个基站过去5分钟，过去10分钟，过去30分钟，过去60分钟，过去所有特征，同时统计未来5分钟，未来10分钟，未来30分钟，未来60分钟的告警与故障特征
2.	使用embedding,将故障与告警进行的先后顺序进行embedding
3.	计算故障的一个历史统计占比，学习关联特征
4.	我们认为如果模型最高预测小于0.5，第三大第四大预测概率相近，并且都大于0.08我们认为其是第三大的故障数据。


运行步骤：
1.	 2.1 feature.py  输出：datafusai_all0905_nodrop.csv 位置：./feature

2.	 2.1feature_label 输出：data_label0905.csv位置：./feature

3.	 2gupeng_fusai_feature.py 输出：gupeng0820.csv位置：./feature

4.	embedding.py输出：alert_embedding.csv，alert_embedding_all.csv位置：./feature

5.	error_embedding.py输出：error_embedding.csv位置：./feature

6.	 3train.py输出：result.csv位置：./out

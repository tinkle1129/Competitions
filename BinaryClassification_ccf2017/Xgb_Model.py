# - * - coding: utf8 - * - -
'''
@ Author : Tinkle G
Creation Time: 2017/11/9
'''
import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import random
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import Data_Read_Write

#记录程序运行时间
import time
start_time = time.time()

#读入数据
train = pd.read_csv("data/train_feature_1_9.csv")
del train['ENDDATE']
print train.columns

print len(train.columns)

test = pd.read_csv("data/test_feature_1_9.csv")
print test.columns


train_xy,val = train_test_split(train, test_size = 0.3,random_state=1)

train_Y = train_xy.TARGET
train_X= train_xy.drop(['TARGET','EID'],axis=1)
test_Y = val.TARGET
test_X = val.drop(['TARGET','EID'],axis=1)
xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_val = xgb.DMatrix(test_X, label=test_Y)

EID = test['EID']
del test['EID']

xg_test = xgb.DMatrix(test)

param={
'booster':'gbtree',
'objective': 'binary:logistic',
'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':8, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':10,
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.01, # 如同学习率
'seed':1000,
'nthread':4,# cpu 线程数
'eval_metric': 'auc'
}

watchlist = [ (xg_train,'train'), (xg_val, 'eval') ]
num_round = 20
bst = xgb.train(param, xg_train, num_round, watchlist)

# get feature importance
features = [x for x in train.columns if x not in ['EID', 'TARGET']]

importance = bst.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv("log/feat_importance.csv", index=False)

plt.figure()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.savefig('log/XGBoost Feature Importance.png')


# get prediction
pred = bst.predict(xg_test)

FORTARGET = []
PROB = []

for i in pred:
    if i<0.5:
        FORTARGET.append(0)
    else:
        FORTARGET.append(1)
    PROB.append(round(i, 4))
ret = {'EID':EID,'FORTARGET':FORTARGET,'PROB':PROB}
Data_Read_Write.__submission(pd.DataFrame(ret))

print np.mean(FORTARGET)

# -*- coding:utf-8 -*-
"""
作者:wesley
日期:2020年12月24日
"""

import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
data = iris.data[:100]
label = iris.target[:100]

train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)

params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],
        'max_depth':4,
        'lambda':10,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':2,
        'eta': 0.1,
        'seed':0,
        'nthread':8,
        'silent':1}
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=10, evals=watchlist)
bst.dump_model('iris_model.txt')
ypred = bst.predict(dtest)
print(ypred)
ypred_output = bst.predict(dtest, output_margin=True)
print(ypred_output)
ypred_leaf = bst.predict(dtest, pred_leaf=True)
print(ypred_leaf)
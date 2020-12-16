# -*- coding:utf-8 -*-
"""
作者:wesley
日期:2020年12月09日
"""


# 导入库
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, ElasticNet# 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from xgboost import XGBRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, mean_absolute_error, \
mean_squared_error, r2_score  # 批量导入指标算法
import matplotlib.pyplot as plt  # 导入图形展示库

#%%

# 数据准备
raw_data = np.loadtxt('regression.txt')  # 读取数据文件
X_raw,y = raw_data[:, :-1],raw_data[:, -1]  # 分割自变量,因变量
model_ss = StandardScaler()
X = model_ss.fit_transform(X_raw)

#%%

# 拆分数据集
num = int(X.shape[0]*0.7)
X_train,X_test = X[:num,:],X[num:,:] # 拆分训练集和测试集
y_train,y_test = y[:num],y[num:] # 拆分训练集和测试集

#%%

# 初选回归模型
n_folds = 5  # 设置交叉检验的次数
model_names = ['BayesianRidge', 'XGBR', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_xgbr = XGBRegressor(random_state=0)  # 建立XGBR对象
model_etc = ElasticNet(random_state=0)  # 建立弹性网络回归模型对象
model_svr = SVR(gamma='scale')  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor(random_state=0)  # 建立梯度增强回归模型对象
model_list = [model_br, model_xgbr, model_etc,model_svr, model_gbr]  # 不同回归模型对象的集合
pre_y_list = [model.fit(X_train, y_train).predict(X_test) for model in model_list]  # 各个回归模型预测的y值列表

#%%

# 模型效果评估
n_samples, n_features = X.shape  # 总样本量,总特征数
model_metrics_functions = [explained_variance_score, mean_absolute_error, mean_squared_error,r2_score]  # 回归评估指标对象集
model_metrics_list = [[m(y_test, pre_y_list[i]) for m in model_metrics_functions] for i in range(len(model_list))]  # 回归评估指标列表
regresstion_score = pd.DataFrame(model_metrics_list, index=model_names,
                   columns=['explained_variance', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print('all samples: %d \t features: %d' % (n_samples, n_features),'\n','-'*60)  # 打印输出样本量和特征数量
print('\n','regression metrics:','\n','-'*60)  # 打印输出标题
print(regresstion_score)  # 打印输出回归指标的数据框

#%%

# 模型效果可视化
plt.figure(figsize=(10, 10))  # 创建画布
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.subplot(len(pre_y_list)+1,1,i+1)
    plt.plot(np.arange(len(y_test)), y_test, color='k', label='true y')  # 画出原始值的曲线
    plt.plot(np.arange(len(y_test)), pre_y_list[i], 'g--', label=model_names[i])  # 画出每条预测结果线
    plt.title('True and {} result comparison'.format(model_names[i]))  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.tight_layout() # 自动调整子图间隔

#%%

# 模型应用
print('regression prediction','\n','-'*40)
new_point_set = [[1.05393, 0., 8.14, 0., 0.538, 5.935, 29.3, 4.4986, 4., 307., 21., 386.85, 6.58],
                 [0.7842, 0., 8.14, 0., 0.538, 5.99, 81.7, 4.2579, 4., 307., 21., 386.75, 14.67],
                 [0.80271, 0., 8.14, 0., 0.538, 5.456, 36.6, 3.7965, 4., 307., 21., 288.99, 11.69],
                 [0.7258, 0., 8.14, 0., 0.538, 5.727, 69.5, 3.7965, 4., 307., 21., 390.95,
                  11.28]]  # 要预测的新数据集
for i, new_point in enumerate(new_point_set):  # 循环读出每个要预测的数据点
    x_matrix = np.array(new_point).reshape(1, -1)
    x_scaled = model_ss.transform(x_matrix)
    new_pre_y = model_xgbr.predict(x_scaled)  # 使用GBR进行预测
    print('predict for new point %d is:  %.2f' % (i + 1, new_pre_y))  # 打印输出每个数据点的预测信息

plt.show()
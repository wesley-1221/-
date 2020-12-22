# -*- coding:utf-8 -*-
"""
作者:wesley
日期:2020年12月22日
"""

# 导入库
import pandas as pd
from sklearn.model_selection import train_test_split                                # 数据分区库
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, \
    precision_score, recall_score, roc_curve                                         # 导入指标库
from imblearn.over_sampling import SMOTE                                            # 过抽样处理库SMOTE
import matplotlib.pyplot as plt
import prettytable                                                                   # 导入表格库
import warnings
from pyecharts.charts import WordCloud
from pyecharts.charts import Pie
from pyecharts import options as opts                                               # 配置项

# 数据预处理
def dataPreprocessing(raw_data):
    X, y = raw_data.iloc[:, :-1], raw_data.iloc[:, -1]                 # 分割X(除最后一列的全部数据),y(最后一列数据)
    # 数据审查
    n_samples, n_features = X.shape                                     # 总样本量(1000行),总特征数(41列)
    print('样本数量: {0}| 特征数量: {1} | 缺失值总数: {2}'.format(n_samples, n_features, raw_data.isnull().any().count()))
    # 填充缺失值
    X = X.fillna(X.mean())
    # 样本均衡处理
    model_smote = SMOTE()                                               # 建立SMOTE模型对象
    X, y = model_smote.fit_sample(X, y)                                 # 输入数据并作过抽样处理
    return X, y
# 切分数据集
def splitDataset(X, y):
    # 拆分数据集
    X = pd.DataFrame(X, columns=raw_data.columns[:-1])                 # raw_data.columns[:-1]第一列，给它当特征
    # 对于随机森林这个模型，它本质上是随机的，设置不同的随机状态（或者不设置random_state参数）可以彻底改变构建的模型
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=0)
    return X_train, X_test, y_train, y_test
# 训练模型
def trainingModel(X_train, y_train, X_test):
    # 树模型对象，条形图高度，显示排序后的最大特征数量，X轴文字，grid不显示网格
    # importance_type=  weight是特征在树中出现的次数，gain是使用特征分裂的平均值增益，cover是作为分裂节点的覆盖的样本比例
    param_dist = {'objective': 'binary:logistic', 'n_estimators': 10,
                  'subsample': 0.8, 'max_depth': 10, 'n_jobs': -1}
    model_xgb = xgb.XGBClassifier(**param_dist)
    model_xgb.fit(X_train, y_train)
    pre_y = model_xgb.predict(X_test)
    return model_xgb, pre_y
# 混淆矩阵
def confusionMatrix(y_test, pre_y):
    # 混淆矩阵表格形式输出
    tp, fn, fp, tn = confusion_matrix(y_test, pre_y).ravel()                                   # 获得混淆矩阵
    confusion_matrix_table = prettytable.PrettyTable(['', 'prediction-1', 'prediction-0'])  # 创建表格实例（好看一些）
    confusion_matrix_table.add_row(['actual-1', tp, fn])  # 增加第一行数据
    confusion_matrix_table.add_row(['actual-0', fp, tn])  # 增加第二行数据
    print('confusion matrix \n', confusion_matrix_table)

    # 混淆矩阵图形输出
    classes = list(set(pre_y))
    # 排序，准确对上分类结果
    classes.sort()
    # 对比，得到混淆矩阵
    confusion = confusion_matrix(y_test, pre_y)
    # 热度图，后面是指定的颜色块，gray也可以，gray_x反色也可以
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # 这个东西就要注意了
    # ticks 这个是坐标轴上的坐标点
    # label 这个是坐标轴的注释说明
    indices = range(len(confusion))
    # 坐标位置放入
    # 第一个是迭代对象，表示坐标的顺序
    # 第二个是坐标显示的数值的数组，第一个表示的其实就是坐标显示数字数组的index，但是记住必须是迭代对象
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    # 就是坐标轴含义说明了
    plt.xlabel('guess')
    plt.ylabel('fact')
    # 显示数据，直观些
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[second_index][first_index])

    # 显示
    plt.show()
# 评价模型指标
def evaluationModel(model_xgb, X_test, y_test, pre_y):
    y_score = model_xgb.predict_proba(X_test)                              # 获得决策树的预测概率
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])                         # ROC曲线
    auc_s = auc(fpr, tpr)                                                  # AUC，roc曲线下的面积
    scores = [round(i(y_test, pre_y), 3) for i in
              (accuracy_score, precision_score, recall_score, f1_score)]   # 列表生成式计算（准确率，精确率，召回率，f1得分值）
    scores.insert(0, auc_s)
    core_metrics = prettytable.PrettyTable()                                         # 创建表格实例
    core_metrics.field_names = ['auc', 'accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
    core_metrics.add_row(scores)                                                      # 增加数据
    print('core metrics\n', core_metrics)
    return y_score
# 可视化
def visualizationPart(model_xgb):
    # 输出特征重要性
    # 树模型对象，条形图高度，显示排序后的最大特征数量，X轴文字，grid不显示网格
    # importance_type=  weight是特征在树中出现的次数，gain是使用特征分裂的平均值增益，cover是作为分裂节点的覆盖的样本比例
    xgb.plot_importance(model_xgb, height=0.5, importance_type='gain', max_num_features=10, xlabel='Gain Split', grid=False)
    plt.show()

    # 输出树形规则图
    # 树模型对象， 树的个数0-9， yes_color为真的线条颜色
    xgb.to_graphviz(model_xgb, num_trees=1, yes_color='#638e5e', no_color='#a40000').view()

    # 获取数据
    importance = model_xgb.get_booster().get_score(importance_type='weight', fmap='')
    tuples = [(k, importance[k]) for k in importance]
    tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
    labels, values = zip(*tuples)

    # 词云
    mywordcloud = WordCloud()
    # 词云图的轮廓也可以选择，有 'circle', 'cardioid', 'diamond', 'triangle-forward', 'triangle', 'pentagon'，默认的词云轮廓为circle
    mywordcloud.add('', tuples, shape='cardioid')
    # 渲染图片
    # 指定渲染图片存放的路径
    mywordcloud.render('词云.html')

    # 环形饼图
    circular_pie_chart = (
        Pie(init_opts=opts.InitOpts(width="1600px", height="1000px"))  # 图形的大小设置
            .add(
            series_name="特征重要性",
            data_pair=[list(z) for z in zip(labels, values)],
            radius=["15%", "50%"],  # 饼图内圈和外圈的大小比例
            center=["30%", "40%"],  # 饼图的位置：左边距和上边距
            label_opts=opts.LabelOpts(is_show=True),  # 显示数据和百分比
        )
            .set_global_opts(legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"))  # 图例在左边和垂直显示
            .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
            ),
        )

    )
    circular_pie_chart.render('环形饼图.html')
# 对于树模型的解释
def explain_tree(X_test, y_score):
    # 前N条规则对应的用户数据
    rule_depth_1 = X_test['internet'] < 0.00284512946             # 第一层级的规则
    rule_depth_2 = X_test['longten'] < 230.75                     # 第二层级的规则
    rule_depth_3 = X_test['total_orders'] < 2.97253799
    rule_depth_4 = X_test['sex'] < 0.972537994
    rule_depth_5 = X_test['wireten'] < 86.0607376
    rule_list = [rule_depth_1, rule_depth_2, rule_depth_3, rule_depth_4, rule_depth_5]
    rule_pd = [pd.DataFrame(i) for i in rule_list]
    rule_pd_merge = pd.concat(rule_pd, axis=1)                    # 建立完整数据框（标记不同层的规则对应到X_test用户是否符合的情况）
    print("数据框\n", rule_pd_merge.head())  # true流失

    warnings.filterwarnings('ignore')
    for i in range(5):
        dyn_rules = rule_pd_merge.iloc[:, :i + 1]                              # 取出top规则
        # print(dyn_rules.values)                                               # 得到一个矩阵(所有true,false)
        dyn_rules['is_true'] = [all(i) == True for i in dyn_rules.values]    # 得到都为true的record
        y_pre_selected = y_score[dyn_rules['is_true']]                         # y_score预测概率
        y_pre_cal = y_pre_selected[:, 1] >= 0.5                                 # 取第二列比较
        total_samples = len(y_pre_cal)                                          # n个条件下的样本量
        is_churn = y_pre_cal.sum()                                              # 流失用户的数量
        churn_rate = float(is_churn) / total_samples                            # 流失用户的占比
        # 计算样本比例
        # total_samples X_test在n个条件下的样本量
        print('-' * 40)
        print('样本量: {}'.format(total_samples))
        print('流失客户: {} | 占比: {:.1%} '.format(is_churn, churn_rate))
        print('没有流失客户: {} | 占比: {:.0%} '.format((total_samples - is_churn), (1 - churn_rate)))
        print('-' * 40)

def main(raw_data):
    X, y = dataPreprocessing(raw_data)
    X_train, X_test, y_train, y_test = splitDataset(X, y)
    model_xgb, pre_y = trainingModel(X_train, y_train, X_test)
    confusionMatrix(y_test, pre_y)
    y_score = evaluationModel(model_xgb, X_test, y_test, pre_y)
    visualizationPart(model_xgb)
    explain_tree(X_test, y_score)

if __name__ == '__main__':
    raw_data = pd.read_csv('classification.csv', delimiter=',')  # 读取数据文件(未加工数据)
    main(raw_data)




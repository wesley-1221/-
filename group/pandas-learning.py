#-*- coding = utf-8 -*-
#@Time : 2020/12/17 10:30
#@Author : ghan
#@File : spider.py

import numpy as np
import pandas as pd

'''通过传入一个numpy的二维数组或者dict对象给pd.DataFrame初始化一个DataFrame对象'''

# 通过numpy二维数组
df1 = pd.DataFrame(np.random.randn(6,6))
print(df1)

# 通过dict字典
df = pd.DataFrame({'A': 1,
                    'B': pd.Timestamp(20201217),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", 'train']),
                    'F': 'foo'})
print(df)

s1 = pd.Series(list("1234"))
# print(s1)
print('*'*7+'*'*7)
print(s1.index)     #索引

print('*'*7+'*'*7)
a = df.C.mean()     #单独计算C列的均值
# print(a)
b = df.describe()   #descibe方法会计算每列数据对象是数值的count, mean, std, min, max, 以及一定比率的值
# print(b)

c = df.C.quantile(0.95) #查看居于95%的值, 默认线性拟合
print(c)

d = df.C.value_counts().head()  #查看C列每个值出现的次数
print(d)

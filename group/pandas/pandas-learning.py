#-*- coding = utf-8 -*-
#@Time : 2020/12/17 10:30
#@Author : ghan
#@File : spider.py

import numpy as np
import pandas as pd

'''通过传入一个numpy的二维数组或者dict对象给pd.DataFrame初始化一个DataFrame对象'''

#通过numpy二维数组创建DataFrame
df1 = pd.DataFrame(np.random.randn(6,6))    #创建一个6行6列的数组
#print(df1)
df2 = pd.DataFrame([['joe','san','wu'],[17,35,12]])
#print(df2)

'''DataFrame分为行索引和列索引，默认情况下是从0开始，也可以自定义索引，
添加行索引使用 index ，添加列索引使用 columns ，此操作称“重置行列索引值”'''

#在创建时更改行索引和列索引（自定义索引）
a1 = pd.DataFrame([['zhang','qi','ling'],[17,18,12]],index=['one','two'],columns=['a','b','c'])
print(a1)

#print(a1.index)      #index设置行索引，
print('*'*7+'*'*7)
#print(a1.columns)    #columns设置列索引


#通过dict字典创建
df = pd.DataFrame({'A':1,
                    'B':pd.Timestamp(20201217),
                    'C': pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D':np.array([3]*4,dtype='int32'),
                    'E':pd.Categorical(["test","train","test",'train']),
                    'F':'foo'})
#print(df)

a = df.C.mean()     #单独计算C列的均值
# print(a)
b = df.describe()   #descibe方法会计算每列数据对象是数值的count, mean, std, min, max, 以及一定比率的值
# print(b)
c = df.C.quantile(0.95) #查看居于95%的值, 默认线性拟合
#print(c)
d = df.C.value_counts().head()  #查看C列每个值出现的次数
#print(d)

print('*'*7+'*'*7)

b1 = df.iloc[:1]        #读取第一行数据
b2 = df.iloc[:,1:3]      #读取第1列到第3列数据
b3 = df.iloc[:,2:]       #读取第2列之后的数据
b4 = df.iloc[:,:3]       #读取前3列数据



fn = pd.DataFrame(np.arange(16).reshape(4,4),index=list("abcd"),columns=list("xwzy"))
print(fn)
#print(fn[0:3])      #行切片 或者 ["a":"d"]
#print('*'*7+'*'*7)
#print(fn.dtypes)    #查看每列的数据类型
print('*'*7+'*'*7)
print(fn[["w","z"]])  #列切片,不能用索引切片，只能用标签切片
print('*'*7+'*'*7)
print(fn.head())    #查看前五行数据（查看后五行数据fn.tail()）


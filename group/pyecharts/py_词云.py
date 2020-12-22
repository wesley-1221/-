# -*- coding:utf-8 -*-
"""
作者:wesley
日期:2020年12月22日
"""

### 使用pyecharts画词云
from pyecharts.charts import WordCloud

data = [('python', 23), ('word', 10), ('cloud', 5)]

mywordcloud = WordCloud()
mywordcloud.add('', data, shape='circle')
### 渲染图片
mywordcloud.render()
### 指定渲染图片存放的路径
### mywordcloud.render('E:/wordcloud.html')
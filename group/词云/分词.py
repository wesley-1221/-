import jieba         #导入库
from wordcloud import WordCloud
excludes = {"发展","坚持","全面","国家","社会","加强","推进"} #排除词
f = open("公报.txt", "r", encoding='utf-8') #打开文件
txt = f.read() #读取文件
f.close() #关闭文件
words = jieba.lcut(txt) #分词
counts = {} #创建计数字典
for word in words: #计数
    if len(word) == 1:
        continue
    else:
        counts[word] = counts.get(word, 0) + 1
for word in excludes: #排除多余字
    del (counts[word])
items = list(counts.items()) #将字典转换成列表
items.sort(key=lambda x: x[1], reverse = True) #排序
print(items)
for i in range(5): #输出前5个
    word, count = items[i]
    print("{0:<10} {1:>5}".format(word,count))
newtxt = ' '.join(words) #用空格分开
print(words)
print(newtxt)
wordcloud = WordCloud(background_color='white', width=800,height= 600,font_path="msyh.ttc",max_font_size=200,min_font_size=80,stopwords=excludes,).generate(newtxt) #颜色，宽度，高度，字体，最大（小）字体，排除词
wordcloud.to_file("公报.png") #生成词云



# coding=utf-8


from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('pollution.csv',  # 读取文件名
                   header=0,  # header代表利用第几行的信息作为列标题
                   index_col=0)  # 此处为第一列用作行索引的列编号或者列名，如果给定一个序列则有多个行索引。
values = dataset.values
print("values:\n\r", values)
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

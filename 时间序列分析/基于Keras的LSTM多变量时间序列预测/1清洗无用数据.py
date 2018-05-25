# coding=utf-8

from pandas import read_csv
from datetime import datetime


# load data
def parsefun(x):
    return datetime.strptime(x, '%Y %m %d %H')


dataset = read_csv('raw.csv',  #读取文件名
                   parse_dates=[['year', 'month', 'day', 'hour']],  # parse_dates用list of lists代表把这些内容合并为时间索引
                   index_col=0,  # 用作行索引的列编号或者列名，如果给定一个序列则有多个行索引。
                   date_parser=parsefun)  # 使用函数来解析日期
dataset.drop('No', axis=1, inplace=True)  # 删除No那一列，axis=1代表删除列，inplace代表是否替换原data
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('pollution.csv')



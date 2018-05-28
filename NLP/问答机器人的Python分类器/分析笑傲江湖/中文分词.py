# -*- coding: utf-8 -*-
# 文本分词，去掉标点符号

import jieba

# 读取文本，进行分词存在新的文件里
fin = open('笑傲江湖.txt', 'r')
fou = open('笑傲江湖_分词.txt', 'w')

line = fin.readline()
while line:
    newline = jieba.cut(line, cut_all=False)
    str_out = " ".join(newline)
    print(type(str_out))
    str_out = str_out.replace('，', '') \
        .replace('。', '').replace('？', '').replace('！', '').replace('“', '') \
        .replace('”', '').replace('\"', '').replace('\'', '').replace('：', '') \
        .replace('‘', '').replace('’', '').replace('-', '').replace('——', '') \
        .replace('（', '').replace('）', '').replace('《', '').replace('》', '') \
        .replace('；', '').replace('.', '').replace('、', '').replace('…', '') \
        .replace('.', '').replace(',', '').replace('，', '').replace('?', '')\
        .replace('!', '').replace('<', '').replace('>', '')
    fou.write(str_out)
    line = fin.readline()

fin.close()
fou.close()
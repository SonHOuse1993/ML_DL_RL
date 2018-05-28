# -*- coding: utf-8 -*-
import gensim.models.word2vec as w2v

model_file_name = '笑傲江湖_模型.txt'
model = w2v.Word2Vec.load(model_file_name)

# 计算两个词的相似度/相关程度
y1 = model.similarity("令狐冲", "岳不群")
print("\"令狐冲\", \"岳不群\"的相似度为：", y1,"\n")

# 计算某个词的相关词列表
y2 = model.most_similar("令狐冲", topn=20)  # 20个最相关的
print("和\"令狐冲\"最相关的词有：")
for item in y2:
	print("\t", item[0], item[1])
print("\n")

# 寻找对应关系
# print ' "boy" is to "father" as "girl" is to ...? \n'
# y3 = model.most_similar(['girl', 'father'], ['boy'], topn=3)
print(' "岳灵珊" 跟 "岳不群" 的关系就像 "任盈盈" 跟 ...？')
y3 = model.most_similar(['盈盈', '岳不群'], ['岳灵珊'], topn=3)
for item in y3:
	print("\t", item[0], item[1])
print("\n")

# more_examples = ["he his she", "big bigger bad", "going went being"]
# for example in more_examples:
# 	a, b, x = example.split()
# 	predicted = model.most_similar([x, b], [a])[0][0]
# 	print
# 	"'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
# print
# "--------\n"

# 寻找不合群的词
y4 = model.doesnt_match("岳不群 林平之 任我行 哈哈".split())
print("不合群的词：", y4)


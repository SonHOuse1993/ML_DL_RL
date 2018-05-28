# -*- coding: utf-8 -*-
import gensim.models.word2vec as w2v

model_file_name = '笑傲江湖_模型.txt'
# 模型训练，生成词向量
sentences = w2v.LineSentence('笑傲江湖_分词.txt')
model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)


# 保存模型，以便重用
# # 保存方式
# model.save("text8.model")
# # 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")
model.save(model_file_name)
# # 以一种C语言可以解析的形式存储词向量
# model.save_word2vec_format("text8.model.bin", binary=True)
# # 对应的加载方式
# # model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)


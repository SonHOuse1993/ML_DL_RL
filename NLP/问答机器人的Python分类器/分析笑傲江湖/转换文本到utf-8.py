# -*- coding: utf-8 -*-
# 读取文本，转码后存在新的文件里
fin = open('笑傲江湖.txt', 'r',encoding="GB2312", errors="ignore")
fou = open('笑傲江湖_uft-8.txt', 'w', encoding="utf-8")

line = fin.readline()
while line:
    fou.write(line)
    line = fin.readline()
fin.close()
fou.close()
print("Convert Done!")

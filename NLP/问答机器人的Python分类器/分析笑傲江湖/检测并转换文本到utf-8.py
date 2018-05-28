# -*- coding: utf-8 -*-
# 检测源码格式，如果不是utf8，则进行转换，否则跳过
# https://www.cnblogs.com/WeyneChen/p/6339962.html

import chardet
import codecs


def findEncoding(s):
    file = open(s, mode='rb')
    buf = file.read()
    result = chardet.detect(buf)
    file.close()
    return result['encoding']


def convertEncoding(s):
    isencoding = findEncoding(s)
    if isencoding != 'utf-8' and isencoding != 'ascii':
        print("convert \"%s\" %s to utf-8" % (s, isencoding))
        with open(s, "r", encoding=isencoding, errors="ignore") as sourceFile:
            contents = sourceFile.read()

        with codecs.open(s[:-4]+"_utf-8.txt", "w", "utf-8") as targetFile:
            targetFile.write(contents)
        print("Convert Done!")
    else:
        print("\"%s\" encoding is %s ,there is no need to convert" % (s, isencoding))


# print(findEncoding('笑傲江湖_GB2312.txt'))
convertEncoding('stop_words.txt')

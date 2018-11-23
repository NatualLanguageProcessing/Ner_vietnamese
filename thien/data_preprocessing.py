# import os
# import underthesea
# from docutils.nodes import line
# import pandas as pd
# import re
# import numpy as np
#
#
# data_src = '../data/vlsp2018/raw/VLSP2018-NER-train-Jan14/'
# data_dest = './data_preprocessed/train.txt'
# data_tmp = data_src + 'The thao/23351556.muc'
#
# string1 = 'Hiệu trưởng <ENAMEX TYPE="ORGANIZATION">Trường THPT <ENAMEX TYPE="LOCATION">Phú Điền</ENAMEX></ENAMEX>, <ENAMEX TYPE="LOCATION">Đồng Tháp</ENAMEX> '
# string2 = '<ENAMEX TYPE="LOCATION">Đồng Tháp</ENAMEX>'
# string3 = 'Hiệu trưởng Trường THPT Phú Điền, Đồng Tháp'
#
# tag = {
#     'begin':'<ENAMEX TYPE="',
#     'org':'<ENAMEX TYPE="ORGANIZATION">',
#     'per':'<ENAMEX TYPE="PERSON">',
#     'loc':'<ENAMEX TYPE="LOCATION">',
#     'mis':'<ENAMEX TYPE="MISCELLANEOUS">',
#     'end':'</ENAMEX>'
# }
#
# def find_tag(string):
#     for key, value in tag.items():
#         if string == value:
#             return key, len(value)
#
# def get_entity(string):
#     if not string.startswith(tag['begin']):
#         return None
#     idx = string.find('>')
#     type, len_tag = find_tag(string[:idx+1])
#     idx_e = len(string)- 9
#     entity = string[len_tag:idx_e]
#     for key, value in tag.items():
#         if key == 'begin':
#             continue
#         entity = entity.replace(value, '')
#     return type, entity
#
# def find_entity(string):
#     i_o = string.find(tag['begin'])
#     if i_o == -1:
#         return 'O', string
#     i_c = string.find(tag['end'])
#     i_s_o = string[i_o+10:].find(tag['begin'])
#     if i_s_o == -1 or i_s_o > i_c-i_o-10:
#         type, entity = get_entity(string[i_o:i_c+9])
#         print(type, entity)
#         if i_s_o > i_c-i_o-10:
#             find_entity(string[i_s_o+i_o+10:])
#     else:
#         type, entity = get_entity(string[i_o:i_c + 18])
#         print(type, entity)
#         type2, entity2 = get_entity(string[i_s_o+i_o+10:i_c+9])
#         print(type2, entity2)
#
#         if string[i_c+18:].find(tag['begin']) != -1:
#             find_entity(string[i_c+18:])
#
#
# with open(data_tmp,'r') as f:
#     string = f.read()
#     find_entity(string)
import underthesea


def load_data_train_2016(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_words = len(lines)

        sentences = []
        sentence = []

        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue
            elif line.isspace():
                sentences.append(sentence)
                sentence = []
            else:
                data = line.split('\t')
                if '_' in data[0]:
                    continue
                data[3] = data[3][:-1]
                sentence.append(data)
        return sentences


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, chunking, label in sent]


def sent2tokens(sent):
    return [token for token, postag, chunking, label in sent]


def tokenize(para):
    # split paragraph to sentences
    try:
        sentences = para.split('. ')
    except Exception as e:
        print(sentences)

    # add '.' after split paragraph
    for idx, sent in enumerate(sentences[:-1]):
        sentences[idx] = sent + "."

    # tokenize sentence
    for idx, sent in enumerate(sentences):
        sentences[idx] = underthesea.chunk(sent)
        for i in range(len(sentences[idx])):
            sentences[idx][i] += tuple('O')
    return sentences


# para = "Cậu bạn soái ca sinh năm 2002 này đập tan mọi suy nghĩ rằng những " \
#        "người học giỏi sẽ là mọt sách, lúc nào cũng chỉ biết có học. Dũng khác " \
#        "biệt hoàn toàn, cậu mê game, biết đánh piano, thích tập võ, tập gym và " \
#        "luôn nuôi tham vọng trở thành một chàng trai cao to, 6 múi. Đó là hình " \
#        "tượng mà Phi Dũng hướng đến."
# a = tokenize(para)
# print(a)
# print(sent2tokens(a[0]))
# print("===========================================")
# print(sent2labels(a[0]))
# print("===========================================")
# print(sent2features(a[0]))

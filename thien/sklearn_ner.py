# -*- coding: utf-8 -*-

import underthesea

list_prev = ['đã', 'đang', 'vẫn', 'là', 'làm', 'chỉ', 'các', 'một', 'đến', 'đi', 'tại', 'ở']
list_aft = [',', ')','"','}']
containshyphen = ['-']

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][2]
    features = {
        'bias' : 1.0,
        'word.lower()' : word.lower(),
        'word.islower()': word.islower(),
        #no space and digit
        'word.isanpla()' : word.isalpha(),
        'word.isupper()' :  word.isupper(),
        # Ki tu dau viet hoa
        'word.istitle()' : word.istitle(),
        # Chi chua cac so
        'word.isdigit()' : word.isdigit(),
        # No space
        'word.isalnum' : word.isalnum(),
        'word.containshyphen' : word in containshyphen,
        'postag' : postag,
        'postag[:2]' : postag[:2],
        

    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.islower()': word1.islower(),
            '-1:word.isanpla()' : word.isalpha(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isalnum' : word1.isalnum(),
            '-1:word.containshyphen' : word in containshyphen,
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.inpre' : word1 in list_prev,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.islower()': word1.islower(),
            '+1:word.isanpla()' : word.isalpha(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isalnum' : word1.isalnum(),
            '+1:word.containshyphen' : word in containshyphen,
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.inaft' : word1 in list_aft,
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, chunking, label in sent]


def sent2tokens(sent):
    return [token for token, postag, chunking, label in sent]

# Predict
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
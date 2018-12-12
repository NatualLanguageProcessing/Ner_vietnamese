import underthesea
import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


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


def get_dict(filename):
    _dictionary = dict()
    list_words = list()
    with open(filename, 'r') as f:
        lines = f.readlines()

        for i, word in enumerate(lines):
            word = word.replace('\n', '')
            list_words.append(word)

    list_words = sorted(set(list_words))
    for i, word in enumerate(list_words):
        _dictionary[word] = i
    _dictionary['unknown'] = len(_dictionary)
    _dictionary['PAD'] = len(_dictionary)

    return _dictionary


dictionary = get_dict('data_preprocessed/words.txt')


def word2idx(word):
    if word in dictionary.keys():
        return dictionary[word]
    else:
        return dictionary['unknown']


def sent2idx(sent):
    out = []
    for word in sent:
        out.append(word2idx(word))
    return out


dict_tags = {
    'B-LOC': 1,
    'B-PER': 2,
    'I-PER': 3,
    'B-ORG': 4,
    'B-MISC': 5,
    'I-LOC': 6,
    'I-MISC': 7,
    'I-ORG': 8,
    'O': 0
}
idx2tag = {i: w for w, i in dict_tags.items()}

max_len = 75


def tag2idx(tag):
    return dict_tags[tag]


def idx2tags(sents):
    out = []
    for sent in sents:
        s = []
        for w in sent:
            w_i = np.argmax(w)
            s.append(idx2tag[w_i])
        out.append(s)
    return out


def prepare_data_train_lstm(filename):
    dataset = load_data_train_2016(filename)
    sents = []
    tags = []
    for sent in dataset:
        s = [w[0] for w in sent]
        t = [w[3] for w in sent]

        sents.append(s)
        tags.append(t)

    idx_sents = [sent2idx(sent) for sent in sents]
    X = pad_sequences(maxlen=max_len, sequences=idx_sents, padding="post", value=len(dictionary)-1)

    y = [[tag2idx(t) for t in s] for s in tags]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx("O"))
    y = [to_categorical(i, num_classes=len(dict_tags)) for i in y]

    return X, np.array(y)


def input_lstm_crf(para):
    sentences = nltk.tokenize.sent_tokenize(para)
    sentences = [underthesea.word_tokenize(sent) for sent in sentences]
    idx_sents = [sent2idx(sent) for sent in sentences]
    X = pad_sequences(maxlen=max_len, sequences=idx_sents, padding="post", value=len(dictionary) - 1)
    return X











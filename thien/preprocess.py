import numpy as np
import underthesea
import nltk
import glob
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# setting parameter
max_len = 75
max_len_char = 10


dict_tags = {
    'PAD': 0,
    'B-LOC': 1,
    'B-PER': 2,
    'I-PER': 3,
    'B-ORG': 4,
    'B-MISC': 5,
    'I-LOC': 6,
    'I-MISC': 7,
    'I-ORG': 8,
    'O': 9
}

idx2tag = { i: w for w, i in dict_tags.items()}
# print(idx2tag)


n_tags = len(dict_tags)

paragragh = 'Hôm nay tôi đi học tại trường đại học Bách Khoa Hà Nội. Tôi tên là Nguyễn Tiến Thiện. Về với quê hương Hải Dương.'

# ====================== pos tag =================
def get_dict_pos_tag(paragraph):
    words = underthesea.chunk(paragraph)
    pos_tags = [word[1] for word in words]
    pos_tags = set(pos_tags)
    try:
        with open('data/dict/postag.txt', 'r') as f:
            tags = f.read().split('\n')
            pos_tags.update(tags)
    except FileNotFoundError as e:
        print(e)
    with open('data/dict/postag.txt', 'w') as f:
        pos_tags = sorted(pos_tags)
        for t in pos_tags[:-1]:
            f.write(t + '\n')
        f.write(pos_tags[-1])
    # print(pos_tags)

# get_dict_pos_tag(paragragh)

def load_dict_pos_tag(filename):
    _dict_pos_tags = dict()
    _dict_pos_tags['PAD'] = 0
    _dict_pos_tags['unknown'] = 1
    with open(filename, 'r') as f:
        pos_tags = f.read().split('\n')
        for i, t in enumerate(pos_tags):
            _dict_pos_tags[t] = i + 2

    return _dict_pos_tags

dict_pos_tags = load_dict_pos_tag('data/dict/postag.txt')


def postag2idx(tags):
    out = []
    for t in tags:
        if t in dict_pos_tags:
            out.append(dict_pos_tags[t])
        else:
            out.append(dict_pos_tags['unknown'])
    return out


# ========================WORD==========================
def get_dict_words(folder_path):
    files = glob.glob(folder_path + '*.txt')
    list_words = set()
    for i, filename in enumerate(files):
        print('file:', i, '/', len(files), ':', filename)
        with open(filename, 'r', encoding="utf16") as f:
            text = f.read()
            # get_dict_pos_tag(text)
            words = underthesea.word_tokenize(text)
            list_words.update(words)

    try:
        with open('data/dict/words.txt', 'r') as f:
            words = f.read().split('\n')
            list_words.update(words)
    except FileNotFoundError as e:
        print(e)
    list_words = sorted(list_words)
    with open('data/dict/words.txt', 'w') as f:
        for word in list_words:
            f.write(word + '\n')

    return list_words

# get_dict_words('../data/vntc/10Topics/Train_Full/Doi song/')

def load_dict_words(filename):
    list_words = list()
    with open(filename, 'r') as f:
        lines = f.readlines()

        for i, word in enumerate(lines):
            word = word.replace('\n', '')
            list_words.append(word)

    list_words = sorted(set(list_words))

    _dictionary = dict()
    for i, word in enumerate(list_words):
        _dictionary[word] = i + 2
    _dictionary['unknown'] = 1
    _dictionary['PAD'] = 0

    return _dictionary


dict_words = load_dict_words('data/dict/words.txt')
n_words = len(dict_words)
dict_idx_words = {i: w for w, i in dict_words.items()}


# def word_pos_tag(text):
#     sentences = nltk.sent_tokenize(text)
#     sentences = [underthesea.chunk(sent) for sent in sentences]
#     return sentences


def words2idx(sentence):
    sentence = underthesea.word_tokenize(sentence)
    s = list()
    for word in sentence:
        # word = word[0]
        if word in dict_words.keys():
            s.append(dict_words[word])
        else:
            s.append(dict_words['unknown'])
    return s

def idx2words(sentence):
    out = [dict_idx_words[w] for w in sentence]
    return out


# paragragh = 'Hôm nay tôi đi học tại trường đại học Bách Khoa Hà Nội. Tôi tên là Nguyễn Tiến Thiện. Về với quê hương Hải Dương.'
# sents = word_pos_tag(paragragh)
# print(sents)
# a = words2idx(sents)
# print(a)


# =========================CHARACTER==============================
def get_dict_chars(file_words):
    list_chars = set()
    with open(file_words, 'r') as f:
        words = f.read().split('\n')
        for word in words:
            list_chars.update(word)
    list_chars = sorted(list_chars)
    with open('data/dict/chars.txt', 'w') as f:
        for c in list_chars:
            f.write(c + '\n')

    return list_chars

# get_dict_chars('data/dict/words.txt')


def load_dict_chars(filename):
    _dict_chars = dict()
    _dict_chars['PAD'] = 0
    _dict_chars['unknown'] = 1
    with open(filename, 'r') as f:
        chars = f.read().split('\n')
        for i, c in enumerate(chars):
            _dict_chars[c] = i+2
    return _dict_chars


dict_chars = load_dict_chars('data/dict/chars.txt')
n_chars = len(dict_chars)


def chars2idx(text):
    out = []
    for c in text:
        if c in dict_chars:
            out.append(dict_chars[c])
        else:
            out.append(dict_chars['unknown'])
    return out



# a = chars2idx(paragragh)
# print(a)

# ====================== input lstm char =============
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


def input_lstm_char_train(filename):
    sentences = load_data_train_2016(filename)
    idx_words = []
    idx_pos_tags = []
    idx_chars = []
    idx_tags = []
    for i, sentence in enumerate(sentences):
        sent = []
        for word in sentence:
            word = word[0]
            if word in dict_words:
                sent.append(dict_words[word])
            else:
                sent.append(dict_words['unknown'])
        idx_words.append(sent)
        sent.clear()

        for word in sentence:
            pos_tags = word[1]
            if pos_tags in dict_pos_tags:
                sent.append(dict_pos_tags[pos_tags])
            else:
                sent.append(dict_pos_tags['unknown'])

        idx_pos_tags.append(sent)
        sent.clear()

        sent = ' '.join([word[0] for word in sentence[:-1]])
        sent = sent + sentence[-1][0]
        idx_chars.append(chars2idx(sent))
        sent = None

        idx_tags.append([dict_tags[word[3]] for word in sentence])

    x_words = pad_sequences(maxlen=max_len,
                            sequences=idx_words,
                            value=dict_words['PAD'],
                            padding='post',
                            truncating='post')

    x_postags = pad_sequences(maxlen=max_len,
                              sequences=idx_pos_tags,
                              value=dict_pos_tags['PAD'],
                              padding='post',
                              truncating='post')

    x_chars = pad_sequences(maxlen=max_len * max_len_char,
                            sequences=idx_chars,
                            value=dict_chars['PAD'],
                            padding='post',
                            truncating='post')

    y = pad_sequences(maxlen=max_len,
                      sequences=idx_tags,
                      value=dict_tags['PAD'],
                      padding='post',
                      truncating='post')

    # y = np.array([to_categorical(i, num_classes=n_tags) for i in y])


    # x_words = np.reshape(x_words, [len(x_words), max_len, 1])
    x_chars = np.reshape(x_chars, [len(x_chars), max_len, -1])
    y = np.reshape(y, [len(y), max_len, 1])
    print(x_words.shape)
    print(x_chars.shape)
    print(y.shape)
    # print(y[0])

    return x_words, x_chars, y


# input_lstm_char_train('data/preprocessed/test.txt')


def input_lstm_char(text):
    sentences = nltk.sent_tokenize(text)
    idx_words = [words2idx(s) for s in sentences]
    idx_postags = [postag2idx([word[1] for word in underthesea.chunk(s)]) for s in sentences]
    idx_chars = [chars2idx(s) for s in sentences]
    # print(idx_postags)
    # print(idx_words[0])

    x_words = pad_sequences(maxlen=max_len,
                            sequences=idx_words,
                            value=dict_words['PAD'],
                            padding='post',
                            truncating='post')

    x_postags = pad_sequences(maxlen=max_len,
                            sequences=idx_postags,
                            value=dict_pos_tags['PAD'],
                            padding='post',
                            truncating='post')

    x_chars = pad_sequences(maxlen=max_len * max_len_char,
                            sequences=idx_chars,
                            value=dict_chars['PAD'],
                            padding='post',
                            truncating='post')

    # print(x_words.shape)
    # print(x_postags.shape)
    return x_words, x_chars
# input_lstm_char(paragragh)


# ===================== INPUT LSTM CRF ======================
def input_lstm_crf_train(filename):
    sentences = load_data_train_2016(filename)
    idx_words = []
    # idx_pos_tags = []
    # idx_chars = []
    idx_tags = []
    for i, sentence in enumerate(sentences):
        sent = []
        for word in sentence:
            word = word[0]
            if word in dict_words:
                sent.append(dict_words[word])
            else:
                sent.append(dict_words['unknown'])
        idx_words.append(sent)
        # sent.clear()

        # for word in sentence:
        #     pos_tags = word[1]
        #     if pos_tags in dict_pos_tags:
        #         sent.append(dict_pos_tags[pos_tags])
        #     else:
        #         sent.append(dict_pos_tags['unknown'])
        #
        # idx_pos_tags.append(sent)
        # sent.clear()
        #
        # sent = ' '.join([word[0] for word in sentence[:-1]])
        # sent = sent + sentence[-1][0]
        # idx_chars.append(chars2idx(sent))
        # sent = None

        idx_tags.append([dict_tags[word[3]] for word in sentence])

        # print(idx_words[-1])

    x_words = pad_sequences(maxlen=max_len,
                            sequences=idx_words,
                            value=dict_words['PAD'],
                            padding='post',
                            truncating='post')

    # x_postags = pad_sequences(maxlen=max_len,
    #                           sequences=idx_pos_tags,
    #                           value=dict_pos_tags['PAD'],
    #                           padding='post',
    #                           truncating='post')
    #
    # x_chars = pad_sequences(maxlen=max_len * max_len_char,
    #                         sequences=idx_chars,
    #                         value=dict_chars['PAD'],
    #                         padding='post',
    #                         truncating='post')

    y = pad_sequences(maxlen=max_len,
                      sequences=idx_tags,
                      value=dict_tags['PAD'],
                      padding='post',
                      truncating='post')

    y = np.array([to_categorical(i, num_classes=n_tags) for i in y])


    # x_words = np.reshape(x_words, [len(x_words), max_len, 1])
    # x_chars = np.reshape(x_chars, [len(x_chars), max_len, -1])
    # y = np.reshape(y, [len(y), max_len, 1])
    # print(x_words.shape)
    # print(x_chars.shape)
    # print(y.shape)
    # print(y[0])

    return x_words, y


def input_lstm_crf(text):
    sentences = nltk.sent_tokenize(text)
    idx_words = [words2idx(s) for s in sentences]
    # idx_postags = [postag2idx([word[1] for word in underthesea.chunk(s)]) for s in sentences]
    # idx_chars = [chars2idx(s) for s in sentences]
    # print(idx_postags)
    # print(idx_words[0])

    x_words = pad_sequences(maxlen=max_len,
                            sequences=idx_words,
                            value=dict_words['PAD'],
                            padding='post',
                            truncating='post')

    # x_postags = pad_sequences(maxlen=max_len,
    #                         sequences=idx_postags,
    #                         value=dict_pos_tags['PAD'],
    #                         padding='post',
    #                         truncating='post')

    # x_chars = pad_sequences(maxlen=max_len * max_len_char,
    #                         sequences=idx_chars,
    #                         value=dict_chars['PAD'],
    #                         padding='post',
    #                         truncating='post')

    # print(x_words.shape)
    # print(x_postags.shape)
    return x_words



# input_lstm_crf_train('data/preprocessed/dev.txt')


# ======================== parse output tag =======================
def output_lstm_crf(predict):
    out = []
    for sent in predict:
        out_i = []
        for w in sent:
            t = np.argmax(w)
            out_i.append(idx2tag[t].replace("PAD", "O"))
        out.append(out_i)
    return out
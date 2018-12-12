import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input, load_model, model_from_json
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pickle


max_len = 75


def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_words = len(lines)
        # print(num_words)

        words = list()
        pos_tags = list()
        chunk_tags = list()
        ner_tags = list()

        sentence_number = []
        sentence = 1

        for i, line in enumerate(lines):
            if line.startswith('#'):
                continue
            elif line.isspace():
                sentence += 1
            else:
                line = line.replace('\n', '')
                data = line.split('\t')
                if '_' in data[0]:
                    continue
                sentence_number.append(sentence)
                words.append(data[0])
                pos_tags.append(data[1])
                chunk_tags.append(data[2])
                ner_tags.append(data[3])


        dict_data = {'sentence': sentence_number, 'words': words, 'pos_tags': pos_tags, 'chunk_tags':chunk_tags, 'ner_tags':ner_tags}
        df_data = pd.DataFrame(dict_data)
        return df_data

df_data = load_data('data_preprocessed/train.txt')
# print(df_data.head(50))

words = list(set(df_data["words"].values))
words.append("ENDPAD")
n_words = len(words)
print(n_words)

tags = list(set(df_data["ner_tags"].values))
n_tags = len(tags)
print(tags)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["words"].values.tolist(),
                                                           s["pos_tags"].values.tolist(),
                                                           s["ner_tags"].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.sentences[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(df_data)
sent = getter.get_next()
sentences = getter.sentences


word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

# print(tag2idx)

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)


y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)


input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

# model.summary()

# history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)
# #
# save_load_utils.save_all_weights(model, 'lstm_crf.model', include_optimizer=False)
#
# hist = pd.DataFrame(history.history)
#
#
# plt.style.use("ggplot")
# plt.figure(figsize=(12,12))
# plt.plot(hist["acc"])
# plt.plot(hist["val_acc"])
# plt.show()


save_load_utils.load_all_weights(model, 'lstm_crf.model')

test_pred = model.predict(X_te, verbose=2)

idx2tag = {i: w for w, i in tag2idx.items()}
# print(idx2tag)
print(test_pred)


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)
print(pred_labels[1])
print(test_labels[1])
# print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
# report = classification_report(y_true=test_labels, y_pred=pred_labels)
# print(report)
# print(type(report))

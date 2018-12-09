from itertools import chain
import pycrfsuite
import underthesea
import os
import data_preprocessing
from preprocess import n_tags, n_words, max_len, input_lstm_crf, input_lstm_crf_train, output_lstm_crf, idx2words
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
import tensorflow as tf

class CRF_Ner(object):
    def __init__(self, path_model="model_saved/test.crfsuite"):
        self.path_model = path_model
        self.tagger = pycrfsuite.Tagger()
        if os.path.exists(path_model):
            self.tagger.open(path_model)

    def fit(self, x, y):
        trainer = pycrfsuite.Trainer()
        for xseq, yseq in zip(x, y):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 100,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train(self.path_model)

    def predict(self, para):
        tokenizes = data_preprocessing.tokenize(para)
        features = [data_preprocessing.sent2features(s) for s in tokenizes]
        tokenizes = [data_preprocessing.sent2tokens(s) for s in tokenizes]
        tokenizes = [t for s in tokenizes for t in s]
        results = list()
        for sent in features:
            results.extend(self.tagger.tag(sent))
        return tokenizes, results

    def get_entity(self, para):
        tokenizes, tagges = self.predict(para)
        words = list()
        for i, tag in enumerate(tagges):
            if tag.startswith('B'):
                for j in range(i+1, len(tagges)):
                    if tagges[j].startswith('O'):
                        # print(tokenizes[i:j])
                        word = (" ".join(tokenizes[i:j]), i, j, tag[2:])
                        words.append(word)
                        break
        return words, tokenizes

para = "Cậu bạn soái ca sinh năm 2002 này đập tan mọi suy nghĩ rằng những " \
       "người học giỏi sẽ là mọt sách, lúc nào cũng chỉ biết có học. Dũng khác " \
       "biệt hoàn toàn, cậu mê game, biết đánh piano, thích tập võ, tập gym và " \
       "luôn nuôi tham vọng trở thành một chàng trai cao to, 6 múi. Đó là hình " \
       "tượng mà Phi Dũng hướng đến."

# print(para)
# crf = CRF_Ner()
# print(crf.get_entity(para))


class LSTM_CRF(object):
    def __init__(self, load_model=True):
        self.model = self._build_model()
        if load_model and os.path.exists('model_saved/lstm_crf.model'):
            save_load_utils.load_all_weights(self.model, 'model_saved/lstm_crf.model')

    def _build_model(self):
        input = Input(shape=(max_len,))
        model = Embedding(input_dim=n_words,
                          output_dim=20,
                          input_length=max_len,
                          mask_zero=True)(input) # 20 dim embedding
        model = Bidirectional(
            LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model) # variational biLSTM

        model = TimeDistributed(Dense(50, activation="relu"))(model)
        crf = CRF(n_tags)
        out = crf(model)

        model = Model(input, out)
        model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
        # model.summary()
        self.graph = tf.get_default_graph()
        return model

    def fit(self, x_train, y_train, x_val, y_val):
        history = self.model.fit(x=x_train, y=y_train,
                                 validation_data=(x_val, y_val),
                                 batch_size=256, epochs=5, verbose=1)
        hist = pd.DataFrame(history.history)
        plt.style.use("ggplot")
        plt.figure(figsize=(12, 12))
        plt.plot(hist["acc"])
        plt.plot(hist["val_acc"])
        plt.show()

        save_load_utils.save_all_weights(self.model, 'model_saved/lstm_crf.model', include_optimizer=False)

        return history.history

    def predict(self, x):
        with self.graph.as_default():
            return self.model.predict(x)

    def get_ner(self, text):
        x = input_lstm_crf(text)
        tokenizes = list()
        for s in x:
            i_pad = np.where(s == 0)
            if len(i_pad[0]) > 0:
            # print(i_pad)
                s = s[:i_pad[0][0]]
            tokenizes.extend(idx2words(s))
        pred = self.predict(x)
        pred = output_lstm_crf(pred)
        pred = [w for sent in pred for w in sent]
        words = list()
        for i, tag in enumerate(pred):
            if tag.startswith('B'):
                for j in range(i+1, len(pred)):
                    if pred[j].startswith('O'):
                        # print(tokenizes[i:j])
                        word = (" ".join(tokenizes[i:j]), i, j, tag[2:])
                        words.append(word)
                        break
        return words, tokenizes


# para = "Hiếm có nơi nào như tại TP Hồ Chí Minh, cách trung tâm quận 1 sầm uất bởi con sông là bán đảo Thanh Đa (quận Bình Thạnh) im lìm với những cánh đồng, ao đầm bỏ hoang suốt 26 năm qua. Cùng chung cảnh ngộ là hàng trăm héc ta đất được thu hồi để làm dự án Khu đô thị Sing Việt thuộc huyện Bình Chánh cũng bị bỏ hoang gần 20 năm nay. Bán đảo Thanh Đa và Khu đô thị Sing Việt đã khiến hàng nghìn người dân bị ảnh hưởng bởi độ “treo” lịch sử bậc nhất tại TP Hồ Chí Minh."
# lstm_crf = LSTM_CRF(load_model=True)
# pred = lstm_crf.get_ner(para)
# print(pred)

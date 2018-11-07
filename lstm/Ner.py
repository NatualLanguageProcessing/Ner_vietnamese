# author: https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

import pandas as pd
import numpy as np

data = pd.read_csv("../dataset/kaggle_en/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

words = list(set(data["Word"].values))
words.append("ENDPAD")

tags = list(set(data["Tag"].values))

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)
sentences = getter.sentences


import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.hist([len(s) for s in sentences], bins=50)
plt.show()


# dictionaries of words and tags
max_len =50
n_words = len(words)
n_tags = len(tags)
word2idx = {w: i for (i,w) in enumerate(words)}
tag2idx = {t: i for (i,t) in enumerate(tags)}

# Map sentences to sequent of number then padding the sequence
from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s ] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding='post', value=n_words - 1)
y = [[tag2idx[w[2]] for w in s ] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag2idx["O"])

from keras.utils import to_categorical
y =  [to_categorical(i, num_classes=n_tags) for i in y]

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

def trainLSTM(): 
    #  fit a LSTM network with an embedding layer
    from keras.models import Model, Input 
    from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
    
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)
    
    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)
    hist = pd.DataFrame(history.history)
    
    plt.figure(figsize=(12,12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()
    return model
def trainLSTM_CRF():
    from keras.models import Model, Input
    from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
    from keras_contrib.layers import CRF
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
    model.summary()
    history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)
    hist = pd.DataFrame(history.history)
    
    plt.style.use("ggplot")
    plt.figure(figsize=(12,12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()
    
    return model
model = trainLSTM_CRF()


from keras.models import load_model
model = load_model('ner_lstm.h5')
# Save model
model.save('ner_lstm.h5')
model.save('ner_lstn_crf.h5')

# Predict
def predict(i):    
    p = model.predict(np.array([X_te[i]]))
    p = np.argmax(p, axis=-1)
    print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
    for w, pred in zip(X_te[i], p[0]):
        print("{:15}: {}".format(words[w], tags[pred]))
    
predict(2283)
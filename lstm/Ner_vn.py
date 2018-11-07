

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = '../dataset/vlsp2016/train.txt'
def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

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
sentences = load_data(path)

def sent2labels(sent):
    return [label for word , token, postag, label in sent]

def sent2words(sent):
    return [word for word , token, postag, label in sent]
def getwords(sent):
    return [word for word , token, postag, label in sent]

# make words and labels
sen_words = [sent2words(s) for s in sentences]
sen_labels = [sent2labels(s) for s in sentences]

# dictionary
words = []
for s in sen_words:
    for w in s:
        words.append(w)
words = list(set(words))
n_words = len(words)
word2idx = {w: i for (i,w) in enumerate(words)}


tags = []
for s in sen_labels:
    for t in s:
        tags.append(t)
tags = list(set(tags))
n_tags = len(tags)
tag2idx = {t: i for (i,t) in enumerate(tags)}

max_len=50
# Map sentences to sequent of number then padding the sequence
from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s ] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding='post', value=n_words - 1)
y = [[tag2idx[w[3]] for w in s ] for s in sentences]
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
#model = trainLSTM()
#model.save('ner_vn_lstm.h5')
from keras.models import load_model
model = load_model('ner_lstm.h5')

def predict(i):    
    p = model.predict(np.array([X_te[i]]))
    p = np.argmax(p, axis=-1)
    print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
    for w, pred in zip(X_te[i], p[0]):
        print("{:15}: {}".format(words[w], tags[pred]))
    
predict(998)
#words.append(set[getwords(s) for s in sentences])
#words = list(set(sentences[]))
#tags = list(set(y_train))
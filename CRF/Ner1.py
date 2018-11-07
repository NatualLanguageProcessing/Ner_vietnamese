# author: https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/
import pandas as pd
import numpy as np
import nltk

def loadData_NLTK():    
    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
    return train_sents, test_sents

def loadData():
    data = pd.read_csv("../dataset/kaggle_en/ner_dataset.csv",encoding="latin1")
    data = data.fillna(method='ffill')
    words = list(set(data['Word'].values))
    n_words = len(words);
    print('Number of all words: ', n_words)
    return data;
data = loadData();

# Get a sentence and tag
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1;
        self.data = data;
        self.empty = False;
    def get_next(self):
        try:
            s = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.n_sent)]
            self.n_sent +=1
            return s['Word'].values.tolist(), s['POS'].values.tolist(), s['Tag'].values.tolist()
        except:
            self.empty = True
            return None, None, None
getter = SentenceGetter(data)
sent, pos, tag = getter.get_next()

class SentencesGetter(object):
    def __init__(self, data):
        self.n_sent =1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w,p,t) for w,p,t in zip(s['Word'].values.tolist(),
                                                          s['POS'].values.tolist(),
                                                        s['Tag'].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getters = SentencesGetter(data)
sentences = getters.sentences


# Simple Approach
def SimpleMethod(data):
    from sklearn.base import BaseEstimator, TransformerMixin
    class MemoryTagger(BaseEstimator, TransformerMixin):
        def fit(self, X,y):
            voc={}
            self.tags =[]
            for x, t in zip(X,y):
                if t not in self.tags:
                    self.tags.append(t)
                if x in voc:
                    if t in voc[x]:
                        voc[x][t] += 1
                    else:
                        voc[x][t] = 1
                else:
                    voc[x] = {t: 1}
            self.memory = {}
            for k, d in voc.items():
                self.memory[k] = max(d, key=d.get)
        def predict(self, X, y=None):
            return [self.memory.get(x,'O') for x in X]
    
    from sklearn.cross_validation import cross_val_predict
    from sklearn.metrics import classification_report
    
    words = data['Word'].values.tolist()
    tags = data['Tag'].values.tolist()
    pred = cross_val_predict(estimator = MemoryTagger(), X=words, y=tags, cv=5)
    report = classification_report(y_pred = pred,y_true=tags)
    print(report)


# Machine learning Random Forest approach
def RanForest(data):
    from sklearn.cross_validation import cross_val_predict
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    
    words = data['Word'].values.tolist()
    tags = data['Tag'].values.tolist()
    def feature_map(word):
        return np.array([word.istitle(), word.islower(), word.isupper(), len(word), word.isdigit(),  word.isalpha()])
    words = [ feature_map(w) for w in words ]
    pred = cross_val_predict(RandomForestClassifier(n_estimators=20), X=words, y=tags, cv=5)
    report = classification_report(y_pred=pred, y_true=tags)
    print(report)

# features and prepare the dataset.
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# train model
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

from sklearn_crfsuite import CRF
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False)
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
crf.fit(X, y)

# Predict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
x_test = 'please refer to this talk by Alex Gaynor.'
x_test = word_tokenize(x_test)
x_test = pos_tag(x_test)
x_test = list(x_test)
x_test = sent2features(x_test)
print(x_test[6])

from sklearn.externals import joblib
# save the classifier
joblib.dump(crf,'ner_crf_en.joblib')

# load it again
crf = joblib.load('ner_crf_en.joblib')

#pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
#report = flat_classification_report(y_pred=pred, y_true=y)
#print(report)


# show xac suat chuyen tiep tu 1 tag den tag khac , va fearure importance for a certain tag


from itertools import chain
import pycrfsuite
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

data_train_path = './data_preprocessed/train.txt'
data_test_path = './data_preprocessed/test.txt'



def load_data(filename):
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


data_train = load_data(data_train_path)
x_train = [sent2features(s) for s in data_train]
y_train = [sent2labels(s) for s in data_train]

data_test = load_data(data_test_path)
x_test = [sent2features(s) for s in data_test]
y_test = [sent2labels(s) for s in data_test]

# print(data_test)


def train():
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0 ,# coefficient for L1 penalty
        'c2': 1e-3, # coefficient for L2 penalty
        'max_iterations': 100, # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    print(trainer.params())

    trainer.train('test.crfsuite')

    print(trainer.logparser.last_iteration)


def test():
    tagger = pycrfsuite.Tagger()
    tagger.open('test.crfsuite')

    example_sent = data_test[4:6]
    print(' '.join(sent2tokens(example_sent)), end='\n\n')
    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))


def bio_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(y_pred_combined,
                                 y_true_combined,
                                 labels=[class_indices[cls] for cls in tagset],
                                 target_names=tagset,
                                 )

# tagger = pycrfsuite.Tagger()
# tagger.open('test.crfsuite')
# y_pred = [tagger.tag(xseq) for xseq in x_test]
# print(bio_classification_report(y_test, y_pred))




test()
# train()
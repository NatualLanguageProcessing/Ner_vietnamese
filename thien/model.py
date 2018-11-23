from itertools import chain
import pycrfsuite
import underthesea
import os
import data_preprocessing


class CRF(object):
    def __init__(self, path_model="test.crfsuite"):
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


# para = "Cậu bạn soái ca sinh năm 2002 này đập tan mọi suy nghĩ rằng những " \
#        "người học giỏi sẽ là mọt sách, lúc nào cũng chỉ biết có học. Dũng khác " \
#        "biệt hoàn toàn, cậu mê game, biết đánh piano, thích tập võ, tập gym và " \
#        "luôn nuôi tham vọng trở thành một chàng trai cao to, 6 múi. Đó là hình " \
#        "tượng mà Phi Dũng hướng đến."
#
# print(para)
# crf = CRF()
# print(crf.get_entity(para))

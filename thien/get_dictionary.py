import underthesea
import glob
import os

path = 'data/vntc/10Topics/Test_Full/'

files = glob.glob('data/vntc/10Topics/*/*/*.txt')

list_words = set()

#
# for i, filename in enumerate(files):
#     if i < 500:
#         continue
#     if i > 1000:
#         break
#     print('file:', i, '/', len(files))
#     with open(filename, 'r', encoding="utf16") as f:
#         text = f.read()
#         words = underthesea.word_tokenize(text)
#         list_words.update(words)
#
#
# with open('data/list_words2.txt', 'w') as f:
#     for word in list_words:
#         f.write(word+'\n')


list_words = set()

with open('data_preprocessed/dict.txt', 'r') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        list_words.update([line])

with open('data/words.txt', 'r') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        list_words.update([line])

list_words = sorted(list_words)
for word in list_words:
    print(word)



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import numpy as np

MAX_WORD_LENGTH = 50

data_train1 = []
data_train2 = []
data_label = []

# Read the data into a list of strings.
def make_voca(filename):
    data = []
    #with open(filename, 'r', encoding='utf-8') as f:
    with open(filename, 'r') as f:
    # with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, line in enumerate(reader):
            if idx > 0:
                data_train1.append(line[3])
                data_train2.append(line[4])
                data_label.append(int(line[5]))
                data += line[3].split() + line[4].split()
    return data


def sen2vec(pure_data):
    vec_data = []

    for sentence in pure_data:

        w2v = []
        words = sentence.split()

        for i in range(MAX_WORD_LENGTH):
            try:
                w2v.append([dictionary[words[i]]])
            except:
                w2v.append([dictionary['UNK']])

        # [None, 50, 128]
        vec_data.append(w2v)

    return vec_data

filename = 'train.csv'
vocabulary = make_voca(filename)

print('Data size', len(vocabulary))

# # Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


# make data=>index, dictionary, reverse index....
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # data: input data => index , count => each word count, dictionary => word:index, reversed_dictionary
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.

x1 = np.array(sen2vec(data_train1))
x2 = np.array(sen2vec(data_train2))
y = np.array(data_label)

print (x1.shape, x2.shape, y.shape)
np.savez('train_idx.npz', train1=x1, train2=x2, label=y)

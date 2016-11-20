'''Trains a memory network on the bAbI dataset.

References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''

from __future__ import print_function

import re
from functools import reduce

import numpy as np
from keras.layers import Merge, Dropout, Activation, Permute, Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if
            not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


challenges = {
    # sample data to understand the code structure
    'single_supporting_fact_sample': '../data/sample_{}.txt',
    # QA1 with 10,000 samples
    'single_supporting_fact': '../data/qa1_single-supporting-fact_{}.txt',
}
challenge_type = 'single_supporting_fact'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(open(challenge.format('test')))
test_stories = get_stories(open(challenge.format('train')))

vocab = sorted(
    reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

input_mem_encoding = Sequential()
input_mem_encoding.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=story_maxlen))
input_mem_encoding.add(Dropout(0.3))

query_encoding = Sequential()
query_encoding.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
query_encoding.add(Dropout(0.3))

softmax_input_query = Sequential()
softmax_input_query.add(Merge([input_mem_encoding, query_encoding], mode='dot', dot_axes=[2, 2]))
softmax_input_query.add(Activation('softmax'))

input_sm_encoding = Sequential()
input_sm_encoding.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=story_maxlen))
input_sm_encoding.add(Dropout(0.3))
input_sm_encoding.add(Permute((2, 1)))

output = Sequential()
output.add(Merge([softmax_input_query, input_sm_encoding], mode='dot', dot_axes=[1, 2]))
output.add(Dropout(0.3))

answer = Sequential()
answer.add(Merge([output, query_encoding], mode='sum'))
# answer.add(Flatten())
answer.add(Dropout(0.3))
answer.add(LSTM(32))
answer.add(Dropout(0.3))
answer.add(Dense(vocab_size, activation='softmax'))

answer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

answer.fit([inputs_train, queries_train, inputs_train], answers_train,
           batch_size=32,
           nb_epoch=120,
           validation_data=([inputs_test, queries_test, inputs_test], answers_test))

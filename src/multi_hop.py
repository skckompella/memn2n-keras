import re

import nltk
import numpy as np
from keras.layers import Activation, Dropout, Input
from keras.models import Model
from MemoryNetworkModel import MemoryRepresentation, OutputLayerW
import glob


def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
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
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    return data


def get_data_vectors(qa_data, qa_vocab):
    stories_one_hot = []
    queries_one_hot = []
    answers_one_hot = []

    story_len = 0

    for stories, query, answer in qa_data:
        stories_one_hot_elem = []
        queries_one_hot_elem = np.zeros(shape=[len(qa_vocab)])
        answers_one_hot_elem = np.zeros(shape=[len(qa_vocab)])
        for story in stories:
            story_one_hot = np.zeros(shape=len(qa_vocab))
            for index, word in enumerate(qa_vocab):
                if word in story:
                    story_one_hot[index] = 1
            stories_one_hot_elem.append(story_one_hot)
        story_len = max(len(stories), story_len)

        for index, word in enumerate(qa_vocab):
            if word in query:
                queries_one_hot_elem[index] = 1
            if word == answer:
                answers_one_hot_elem[index] = 1

        stories_one_hot.append(np.array(stories_one_hot_elem))
        queries_one_hot.append(queries_one_hot_elem)
        answers_one_hot.append(answers_one_hot_elem)

    return np.array(stories_one_hot), np.array(queries_one_hot), np.array(answers_one_hot), story_len


def pad_stories(stories_data, vocab, story_maxlen):
    padded_stories = []
    for stories in stories_data:
        padded_stories_elem = []
        for index in xrange(story_maxlen - len(stories)):
            padded_stories_elem.append(np.zeros(len(vocab)))
        for index in xrange(len(stories)):
            padded_stories_elem.append(stories[index])
        padded_stories.append(np.array(padded_stories_elem))

    return np.array(padded_stories)


def main():

    data_folder = '../data/en/'
    data_folder_10k = '../data/en-10k/'

    qa_task = 1
    num_hops = 2

    for qa_task in range(1, 21):
        print '############################## Task: ', qa_task, '##############################'

        train_file = glob.glob(data_folder_10k + 'qa' + str(qa_task) + '_*_train.txt')[0]
        test_file = glob.glob(data_folder + 'qa' + str(qa_task) + '_*_test.txt')[0]

        train_data = get_stories(open(train_file))
        test_data = get_stories(open(test_file))
        vocab = set(nltk.wordpunct_tokenize(open(train_file).read()))
        test_vocab = set(nltk.wordpunct_tokenize(open(test_file).read()))
        vocab.update(test_vocab)

        train_stories, train_queries, train_answers, train_story_len = get_data_vectors(train_data, vocab)
        test_stories, test_queries, test_answers, test_story_len = get_data_vectors(test_data, vocab)

        story_maxlen = max(train_story_len, test_story_len)
        train_stories = pad_stories(train_stories, vocab, story_maxlen)
        test_stories = pad_stories(test_stories, vocab, story_maxlen)

        for num_hops in xrange(2, 4):

            print 'Hop: ', num_hops

            story_m_input = Input(shape=(story_maxlen, len(vocab)), name='input_m')
            query_u_input = Input(shape=[len(vocab), ], name='query_u')

            output = MemoryRepresentation(output_dim=20, num_hops=num_hops)([story_m_input, query_u_input])

            answer = OutputLayerW(output_dim=len(vocab))(output)
            answer = Activation('softmax')(answer)

            qa_model = Model(input=[story_m_input, query_u_input], output=answer)
            qa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

            qa_model.fit([train_stories, train_queries], train_answers, batch_size=1, nb_epoch=120,
                         validation_data=([test_stories, test_queries], test_answers))

        print '####################################################################################'


if __name__ == '__main__':
    main()

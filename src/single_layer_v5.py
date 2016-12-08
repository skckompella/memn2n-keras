import re

import keras.backend as K
import nltk
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Activation, Dropout, Input
from keras.models import Model
import glob


class MemoryRepresentation(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MemoryRepresentation, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][2]
        initial_A_value = np.random.uniform(0, 1, size=[self.output_dim, input_dim])
        initial_B_value = np.random.uniform(0, 1, size=[self.output_dim, input_dim])
        initial_C_value = np.random.uniform(0, 1, size=[self.output_dim, input_dim])
        self.input_dim = input_shape[0][1]
        self.A = K.variable(initial_A_value)
        self.B = K.variable(initial_B_value)
        # self.C = K.variable(initial_C_value)
        self.trainable_weights = [self.A, self.B]
        super(MemoryRepresentation, self).build(input_shape)

    def call(self, inputs, mask=None):
        input_a = inputs[0]
        input_c = inputs[0]
        input_b = inputs[1]
        mem_m_tensor = K.dot(input_a, K.transpose(self.A))
        # mem_c_tensor = K.dot(input_c, K.transpose(self.C))
        query_tensor = K.dot(input_b, K.transpose(self.B))

        softmax_tensor = K.softmax(K.reshape(K.dot(mem_m_tensor, K.transpose(query_tensor)), (1, self.input_dim)))
        output_tensor = K.reshape(K.dot(softmax_tensor, mem_m_tensor), (1, self.output_dim))
        return output_tensor

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


class OutputLayerW(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(OutputLayerW, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.uniform(0, 1, size=[input_dim, self.output_dim])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]
        super(OutputLayerW, self).build(input_shape)

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


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

    qa_task = 2
    train_file = glob.glob(data_folder_10k + 'qa' + str(qa_task) + '_*_train.txt')[0]
    test_file = glob.glob(data_folder + 'qa' + str(qa_task) + '_*_test.txt')[0]

    # print train_file
    # print test_file
    # exit()

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

    story_m_input = Input(shape=(story_maxlen, len(vocab)), name='input_m')
    query_u_input = Input(shape=[len(vocab), ], name='query_u')

    output = MemoryRepresentation(output_dim=20)([story_m_input, query_u_input])

    answer = OutputLayerW(output_dim=len(vocab))(output)
    answer = Activation('softmax')(answer)

    qa_model = Model(input=[story_m_input, query_u_input], output=answer)
    qa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    qa_model.fit([train_stories, train_queries], train_answers, batch_size=1, nb_epoch=120,
                 validation_data=([test_stories, test_queries], test_answers))


if __name__ == '__main__':
    main()

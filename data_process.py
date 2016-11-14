
import numpy as np
import re
import os

PATH_TO_DATA = '/Users/skc/Projects/memn2n-keras/tasks_1-20_v1-2/en'
TRAIN_DIR = PATH_TO_DATA + '/TRAIN'
TEST_DIR = PATH_TO_DATA + '/TEST'

def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*]+|[^a-zA-Z\d\s]+", sentence)


def parse_stories(lines,  only_supporting=False):
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
    data = parse_stories(f.readlines())
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data
    pass

def get_stats():
    filenames = os.listdir(TRAIN_DIR)
    train_stories = []
    test_stories = []
    for f in filenames:
        train_stories += get_stories(open(os.path.join(TRAIN_DIR, f), 'r'))

    filenames = os.listdir(TEST_DIR)
    for f in filenames:
        test_stories += get_stories(open(os.path.join(TEST_DIR, f), 'r'))

    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))

    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')

get_stats()
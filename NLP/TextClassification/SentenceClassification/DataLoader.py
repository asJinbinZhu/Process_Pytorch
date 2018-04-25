import torch
from torch.autograd import Variable
import torch.utils.data as Data

SEED = 1

import codecs
import random


def load_MR_data():
    file_pos = './datasets/MR/rt-polarity.pos'
    file_eng = './datasets/MR/rt-polarity.neg'
    #print('Load MR data from ', file_pos, ' and ', file_eng)

    pos_sents = codecs.open(file_pos, 'r', 'utf8').read().split('\n')
    neg_sents = codecs.open(file_eng, 'r', 'utf8').read().split('\n')

    random.seed(SEED)
    random.shuffle(pos_sents)
    random.shuffle(neg_sents)
    #print(len(pos_sents), len(neg_sents))

    #train_data = [(sent, 1) for sent in pos_sents[: 4250]] + [(sent, 0) for sent in neg_sents[: 4250]]
    #dev_data = [(sent, 1) for sent in pos_sents[4250:4800]] + [(sent, 0) for sent in neg_sents[4250:4800]]
    #test_data = [(sent, 1) for sent in pos_sents[4800:]] + [(sent, 0) for sent in neg_sents[4800:]]

    train_data = [(sent, 1) for sent in pos_sents[: 1]] + [(sent, 0) for sent in neg_sents[: 1]]
    dev_data = [(sent, 1) for sent in pos_sents[1:2]] + [(sent, 0) for sent in neg_sents[1:2]]
    test_data = [(sent, 1) for sent in pos_sents[2:3]] + [(sent, 0) for sent in neg_sents[2:3]]
    #print('train_data: ', len(train_data), 'dev_data: ', len(dev_data), 'test_data: ', len(test_data))
    #print(train_data, '\n', dev_data, '\n', test_data, '\n')
    '''
    train_data:  2 dev_data:  2 test_data:  2
    [("it's something of the ultimate scorsese film , with all the stomach-turning violence , colorful new york gang lore and other hallmarks of his personal cinema painted on their largest-ever historical canvas . ", 1), ("the problem isn't that the movie hits so close to home so much as that it hits close to home while engaging in such silliness as that snake-down-the-throat business and the inevitable shot of schwarzenegger outrunning a fireball . ", 0)] 
    [("a savvy exploration of paranoia and insecurity in america's culture of fear . ", 1), ('a tedious parable about honesty and good sportsmanship . ', 0)] 
    [('the off-center humor is a constant , and the ensemble gives it a buoyant delivery . ', 1), ("schmaltzy and unfunny , adam sandler's cartoon about hanukkah is numbingly bad , little nicky bad , 10 worst list bad . ", 0)] 
    '''

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    word_to_idx = build_token_to_idx([s for s, _ in train_data+dev_data+test_data])
    label_to_idx = {0:0, 1:1}
    #print('size of vocab: ', len(word_to_idx), 'size of label: ', len(label_to_idx))

    return train_data, dev_data, test_data, word_to_idx, label_to_idx

def build_token_to_idx(sentence):
    token_to_idx = dict()
    #print(len(sentence))
    for sent in sentence:
        for token in sent.split(' '):
            if token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)
    token_to_idx['<pad>'] = len(token_to_idx)
    return token_to_idx

def prepare_sentence(seq, _to_idx):
    return Variable(torch.LongTensor([_to_idx[w] for w in seq.split(' ')]))
def prepare_label(label, label_to_idx):
    return Variable(torch.LongTensor([label_to_idx[label]]))

load_MR_data()
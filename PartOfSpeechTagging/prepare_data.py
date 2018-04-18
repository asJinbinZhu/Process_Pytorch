import torch
from torch.autograd import Variable

# prepare data
def prepare_sequence(seq,to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

training_data = [
    ('The dog ate the apple'.split(),['DET','NN','V','DET','NN']),
    ('Everybody read that book'.split(),['NN','V','DET','NN'])
]

word_to_ix = {}

for sent,tags in training_data:
    for word in sent:
        if word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

print(word_to_ix)
tag_to_ix = {'DET':0,'NN':1,'V':2}
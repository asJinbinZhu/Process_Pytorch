import torch
from torch.autograd import Variable

inputs = [
    (("The dog ate the apple").split(), ["DET", "NN", "V", "DET", "NN"]),
    (("Everybody read the book").split(), ["NN", "V", "DET", "NN"])
]

'''
******************************1. word_to_idx**********************************************
'''
word_to_idx = {}
tag_to_idx = {}

for words, tags in inputs:
    for word in words:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for tag in tags:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)

dim_word = 10
dim_tag = 3
word_embeds = torch.nn.Embedding(len(word_to_idx), dim_word)
tag_embeds = torch.nn.Embedding(len(tag_to_idx), dim_tag)


def make_sentence(sentence, wordOrtag_to_idx):
    sentence = [wordOrtag_to_idx[word_or_tag] for word_or_tag in sentence]
    sentence = Variable(torch.LongTensor(sentence))
    return sentence

test_data = [
    (("Everybody ate the apple").split(), ["NN", "V", "DET", "NN"]),
    (("Everybody ate the apple").split(), ["NN", "V", "DET", "NN"])
]

for data in test_data:
    sentence, tag = data
    sentence = make_sentence(sentence, word_to_idx)
    tag = make_sentence(tag, tag_to_idx)

    '''
    ******************************2. embedding**********************************************
    '''
    print(word_embeds(sentence))
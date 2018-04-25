from __future__ import unicode_literals, print_function, division
from io import open
import glob

def find_files(path):
    return glob.glob(path)

# print(find_files('data/names/*.txt'))

import  unicodedata
import string

all_letters = string.ascii_letters + " .,'"
n_letters = len(all_letters)

def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(unicode2ascii('Ślusàrski'))   # Slusarski

category_lines = {}
all_categories = []

def read_lines(file_name):
    lines = open(file_name, encoding = 'utf-8').read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]

for file_name in find_files('data/names/*.txt'):
    category = file_name.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = read_lines(file_name)
    category_lines[category] = lines

n_categories = len(all_categories)

# print(category_lines['Italian'][:5])

import torch
def letter2idx(letter):
    return all_letters.find(letter)

def letter2tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter2idx(letter)] = 1
    return tensor

def line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter2idx(letter)] = 1
    return  tensor

# print(letter2tensor('J'))
# print(line2tensor('Jones').size())

from torch.autograd import Variable

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        #self.lstm = torch.nn.LSTM(input_size, hidden_size)

        self.i2h = torch.nn.Linear(input_size+hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size+hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)

        #hidden = self.init_hidden()
        #output, hidden = self.lstm(input, hidden)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    def init_hidden(self):
        '''
        return ((Variable(torch.zeros(1, 1, self.hidden_size))),
                 (Variable(torch.zeros(1, 1, self.hidden_size))))
                 '''
        return Variable(torch.zeros(1, 1))

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


input = Variable(letter2tensor('A'))
hidden = Variable(torch.zeros(1, n_hidden))

#out, nexthidden = rnn(input.view(1,1,-1))
out, nexthidden = rnn(input, hidden)
print(out, nexthidden)

'''
input = Variable(line2tensor('Albert'))
#hidden = Variable(torch.zeros(1, n_hidden))
#out, next_hidden = rnn(input[0], hidden)
out, next_hidden = rnn(input[0])
#print(out, next_hidden)
'''

def category2output(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i
print(category2output(out))


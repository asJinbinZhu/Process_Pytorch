import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(1)

# training data
inputs = [Variable(torch.randn(1,3)) for _ in range(5)]

# net
lstm = torch.nn.LSTM(3,3)

# initialize data
hidden = (
    Variable(torch.randn(1,1,3)),
    Variable(torch.randn((1,1,3)))
)
'''
for i in inputs:
    out, hidden = lstm(i.view(1,1,-1),hidden)
    print('out: ', out, 'hidden: ', hidden)
'''
input = torch.cat(inputs).view(len(inputs),1,-1)
out, hidden = lstm(input, hidden)
# print('out: ', out, 'hidden: ', hidden)

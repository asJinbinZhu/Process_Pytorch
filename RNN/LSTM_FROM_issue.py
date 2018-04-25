import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

def decoder(input_, embedding, lstm, projection, states):
    """ unroll the LSTM Cell, returns the flattened logits"""
    emb = embedding(input_.t())
    hs = []
    for i in range(input_.size(1)):
        h, c = lstm(emb[i], states)
        hs.append(h)
        states = (h, c)
    lstm_out = torch.stack(hs, dim=0)
    logit = projection(lstm_out.contiguous().view(-1, lstm.hidden_size))
    return logit

embedding = nn.Embedding(4, 64, padding_idx=0)
lstm = nn.LSTMCell(64, 64,)
projection = nn.Linear(64, 4)

input_ = Variable(torch.LongTensor([[1, 2, 3], [3, 2, 1]]))
states = (Variable(torch.zeros(2, 64)), Variable(torch.zeros(2, 64)))
target = Variable(torch.LongTensor([[3, 2, 1], [2, 3, 1]]))

logit = decoder(input_, embedding, lstm, projection, states)
loss = F.cross_entropy(logit, target.t().contiguous().view(-1))
loss.backward()  #  RuntimeError: No grad accumulator for a saved leaf!
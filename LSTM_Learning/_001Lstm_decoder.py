import torch
from torch.autograd import Variable
import torch.nn.functional as F
#import _000HookHepler as hook_helper

def handle_variable_predict_hook(grade):
    print('***********predict_hook***************')
    print('grade: ', grade)
    # modify
    # grade.data[0] = 0
    print('**************************')

def decoder(imput_, embedding, lstm, projection, stages):
    emb = embedding(imput_.t())
    hs = []
    for i in range(input_.size(1)):
        h, c = lstm(emb[i], stages)
        h.register_hook(handle_variable_predict_hook)
        hs.append(h)
        stages = (h, c)
    lstm_out = torch.stack(hs, dim=0)
    logit = projection(lstm_out.contiguous().view(-1, lstm.hidden_size))
    return logit

embedding = torch.nn.Embedding(4, 64, padding_idx=0)
lstm = torch.nn.LSTMCell(64, 64)

projection = torch.nn.Linear(64, 4)

input_ = Variable(torch.LongTensor([[1, 2, 3], [3, 2, 1]]))
stages = (
    Variable(torch.zeros(2, 64)),
    Variable(torch.zeros(2, 64))
)
targets = Variable(torch.LongTensor([[3, 2, 1], [1, 2, 3]]))

logit = decoder(input_, embedding, lstm, projection, stages)
loss = F.cross_entropy(logit, targets.t().contiguous().view(-1))
loss.backward()
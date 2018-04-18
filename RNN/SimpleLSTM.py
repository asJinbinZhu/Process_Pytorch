import torch
from torch.autograd import Variable

torch.manual_seed(1)

time_steps = 2
batch_size = 1
in_size = 2
classes_no = 3

input_seq = Variable(torch.randn(time_steps,batch_size,in_size))
target = Variable(torch.LongTensor(batch_size).random_(0,classes_no-1))

class LSTMTagger(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_nums):
        super(LSTMTagger, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_nums = layer_nums

        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_nums)

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, classes_no)),
                Variable(torch.zeros(1, 1, classes_no)))

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        last_out = out[-1]
        return last_out

model = LSTMTagger(in_size, classes_no, 1)

print(model)
params = model.state_dict()
for k,v in params.items():
    print(k, v)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

hidden = model.init_hidden()
print('hidden: ',hidden[0])
for epoch in range(1):
    out= model(input_seq, hidden)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
params = model.state_dict()
for k,v in params.items():
    print(k, v)
'''
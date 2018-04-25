import torch
from torch.autograd import Variable

class LSTMHandle(torch.nn.Module):
    def __init__(self):
        super(LSTMHandle, self).__init__()
        self.lstm = torch.nn.LSTM(3, 4, num_layers=2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            (Variable(torch.randn(1, 1, 3))),
            (Variable(torch.randn(1, 1, 3)))
        )

    def forward(self, input):
        hidden = self.hidden()
        out, hidden =  self.lstm(input, hidden)
        return out, hidden


lstm = LSTMHandle()

params = lstm.state_dict()
for k,v in params.items():
    print(k,v)


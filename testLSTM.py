import torch
from torch.autograd import Variable

class RNN(torch.nn.Module):
    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(RNN,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.lstm = torch.nn.LSTM(
            in_dim,
            hidden_dim,
            n_layer,
            batch_first=True,
        )
        self.classifier = torch.nn.Linear(hidden_dim,n_class)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.classifier(out)
        return out

net = RNN(2,3,1,2)

print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)
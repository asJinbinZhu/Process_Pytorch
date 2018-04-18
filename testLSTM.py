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

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)

test_data = Variable(torch.randn(2,1))
target_data = Variable(torch.rand(2,1))
for epoch in range(1):
    pre = net(test_data)
    loss = criterion(pre,target_data)
    #print('loss: ',loss)
    #optimizer.zero_grad()
    loss.backward()
    #optimizer.step()

print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)
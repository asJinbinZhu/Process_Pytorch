'''

1. Train a network
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1,1,5),dim=1) # x data(Tensor), shape(100,1)
y= x.pow(2)+0.2*torch.rand(x.size())

x,y = Variable(x),Variable(y)

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        hidden_layer = F.relu(self.hidden(x))
        direct_layer = self.predict(hidden_layer)
        return direct_layer

net = Net(n_features=1,n_hidden=3,n_output=1)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

for epoch in range(500):
    pre = net(x)
    loss = criterion(pre,y)
    #print('loss: ',loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)

torch.save(net.state_dict(),'FullyNN_params.pkl')  # entire net

'''
test_data = Variable(torch.Tensor([[4.0]]))
net2 = torch.load('FullyNN.pkl')
print("Test: ",net2(test_data))
'''


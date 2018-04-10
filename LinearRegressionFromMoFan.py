import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # x data(Tensor), shape(100,1)
y= x.pow(2)+0.2*torch.rand(x.size())

x,y = Variable(x),Variable(y)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(n_features=1,n_hidden=10,n_output=1)
# print(net)
'''
Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
'''
# params = net.state_dict()
# for k,v in params.items():
#     print(k)                #   hidden.weight
                            #   hidden.bias
                            #   predict.weight
                            #   predict.bias
#     print(v)
'''
hidden.weight

-0.9756
-0.0615
 0.1794
 0.0560
-0.3458
-0.2194
 0.6620
-0.3359
 0.1443
-0.5106
[torch.FloatTensor of size 10x1]

hidden.bias

-0.7628
 0.7923
 0.7295
-0.5472
 0.2616
 0.8999
-0.8319
 0.8767
 0.1845
-0.7854
[torch.FloatTensor of size 10]

predict.weight

 0.2925 -0.0547 -0.2520 -0.1223 -0.1374 -0.1963 -0.2200  0.2498 -0.1357  0.2641
[torch.FloatTensor of size 1x10]

predict.bias

1.00000e-02 *
  1.4114
[torch.FloatTensor of size 1]
'''

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

plt.ion()
plt.show()

for epoch in range(100):
    pre = net(x)
    loss = criterion(pre,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # show the training process
    if epoch%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),pre.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

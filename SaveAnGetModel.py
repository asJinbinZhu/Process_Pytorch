import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # x data(Tensor), shape(100,1)
y= x.pow(2)+0.2*torch.rand(x.size())

x,y = Variable(x,requires_grad=False),Variable(y,requires_grad=False)

def save():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

    for epoch in range(100):
        pre = net(x)
        loss = criterion(pre,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),pre.data.numpy(),'r-',lw=5)

    torch.save(net,'net1.pkl')  # entire net
    torch.save(net.state_dict(),'net1.params.pkl') # only parameters


def restore_net():
    net = torch.load('net1.pkl')
    pre = net(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)

def restore_params():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    net.load_state_dict(torch.load('net1.params.pkl'))

    pre = net(x)

    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)
    plt.show()

save()
restore_net()
restore_params()
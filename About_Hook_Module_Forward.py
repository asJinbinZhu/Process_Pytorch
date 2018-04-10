import torch
from torch.autograd import Variable

def handle_hook(module,input,output):
    print(module)
    print('Input',input)
    print('Output',output)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()

    def forward(self, x):
        return x+1

net = Net()
x = Variable(torch.Tensor([3.0]),requires_grad=True)
net.register_forward_hook(handle_hook)
net(x)

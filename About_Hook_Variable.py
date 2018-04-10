import torch
from torch.autograd import Variable

def handle_hook(grad):
    return grad*3

x = Variable(torch.Tensor([0,0,0]),requires_grad=True)
x.register_hook(handle_hook)

x.backward(torch.Tensor([1,1,1]))
print(x.grad)
import torch
from torch.autograd import Variable

torch.manual_seed(1)

'''
Question:
    z = (y1^2 + y2^2)/2
    y = x + 2
'''
if __name__ == '__main__1':

    def changeGrad(grad):
        #grad.data[0] = grad.data[0]-int(grad.data[0]) #get the xiaoshu
        grad.data[0] = 1.0
        grad.data[1] = 1.0
        return grad

    x = Variable(torch.randn(2,1),requires_grad=True)
    print(x)
    x_hook = x.register_hook(changeGrad)

    y = x+2
    z = torch.mean(torch.pow(y,2))
    z.backward()
    print(x.grad.data)

'''
without hook:

Variable containing:
 0.6614
 0.2669
[torch.FloatTensor of size 2x1]


 2.6614
 2.2669
[torch.FloatTensor of size 2x1]
'''


'''
with hook:

Variable containing:
 0.6614
 0.2669
[torch.FloatTensor of size 2x1]


 5.3227
 4.5338
[torch.FloatTensor of size 2x1]
'''

if __name__ == '__main__1':
    import torch
    from torch.autograd import Variable
    import torch.nn.functional as F

    def for_hook(module,input,output):
        print(module)
        for in_val in input:
            print('input val:\n',in_val)
        for out_val in output:
            print('output val:\n',out_val)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model,self).__init__()

        def forward(self, x):
            return x+1

    model = Model()
    x = Variable(torch.FloatTensor([1.0]),requires_grad=True)
    handle = model.register_forward_hook(for_hook)
    print(model(x))
    handle.remove()

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    from torch.nn import Parameter
    import math

    def bh(module,input,output):
        print(module)
        print('Grad Input: ',input)
        print('Grad Output: ',output)
        return input[0]*0,input[1]*0

    class Linear(torch.nn.Module):
        def __init__(self,in_features,out_features,bias=True):
            super(Linear,self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Parameter(torch.Tensor(out_features,in_features))
            if bias:
                self.bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias',None)
            self.reset_parameters()

        def reset_parameters(self):
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv,stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv,stdv)

        def forward(self, input):
            if self.bias is None:
                return self._backend.Linear()(input,self.weight)
            else:
                return self._backend.Linear()(input,self.weight,self.bias)
    x = Variable(torch.FloatTensor([[1,2,3]]),requires_grad=True)
    mod = Linear(3,1,bias=False)
    mod.register_backward_hook(bh)

    out = mod(x)
    out.register_hook(lambda grad: 0.1*grad)
    out.backward()

    print(['*']*20)
    print('x.grad: ',x.grad)
    print(mod.weight.grad)
import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(1)

def handle_forward_hook(module,input,output):
    print('***********forward_hook***************')
    print(module)
    print('Forward Input', input)
    print('Output Output', output)
    print('**************************')

def handle_backward_hook(module,input,output):
    print('***********backward_hook***************')
    print(module)
    print('Grad Input',input)
    print('Grad Output',output)
    #return input[0]*2,input[1]*2,input[2]*2
    '''
    print('Inside '+self.__class__.__name__+' backward')
    print('Inside class: ',self.__class__.__name__)
    print('')
    print('Grad input: ',type(input))
    print('Grad input[0]: ',type(input[0]))
    print('Grad output: ', type(output))
    print('Grad output[0]: ', type(output[0]))
    print('')
    print('Grad input size: ',input[0].size())
    print('Grad output size: ',output[0].size())
    print('Grad input norm: ',input[0].data.norm())
    '''
    print('**************************')

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data = Variable(torch.Tensor([[2.0],[4.0],[6.0]]))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        y_pre = self.linear(x)
        return y_pre
net = Net()
net.register_forward_hook(handle_forward_hook)
net.register_backward_hook(handle_backward_hook)

#----------------------------Modle Structure-----------------------------------

print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)

'''
Net(
  (linear): Linear(in_features=1, out_features=1, bias=True)
)
linear.weight 
 0.1989
[torch.FloatTensor of size 1x1]

linear.bias 
 0.7788
[torch.FloatTensor of size 1]
'''
#----------------------------Modle Structure-----------------------------------

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=1)

for epoch in range(1):
    y_pre = net(x_data)
    loss = criterion(y_pre,y_data)
    print('y_pre: ',y_pre)
    print('y_data: ',y_data)

    print('loss: ',loss)

    optimizer.zero_grad()
    loss.backward()
    print('Grad of linear.weight:\n',net.linear.weight.grad)
    optimizer.step()

test_data = Variable(torch.Tensor([4.0]))
print('After training: \ninput 4.0, output:',net(test_data))
for k,v in params.items():
    print(k,v)


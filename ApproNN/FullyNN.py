'''

1. Train a network
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(1)

def handle_forward_hook(module,input,output):
    print('***********forward_hook***************')
    #print(module)
    print('Forward Input', input)
    print('Output Output', output)
    print('**************************')

def handle_backward_hook(module,input,output):
    print('***********backward_hook***************')
    print(module)
    print('Grad Input',input)
    print('Grad Output',output)
    print('**************************')

def handle_variable_hidden_hook(grade):
    print('***********hidden_hook***************')
    #grade.data[0][0] = 0.0
    #grade.data[0][1] = 0.0
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

def handle_variable_predict_hook(grade):
    print('***********predict_hook***************')
    print('grade: ',grade)
    #grade.data[0][0] = 0.0
    # modify
    #grade.data[0] = 0
    print('**************************')

x = torch.unsqueeze(torch.linspace(-1,1,5),dim=1) # x data(Tensor), shape(100,1)
#print(x)
y= x.pow(2)+0.2*torch.rand(x.size())
#print('y: ', y)

x,y = Variable(x, requires_grad = True),Variable(y)

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        self.hidden.register_forward_hook(handle_forward_hook)
        #self.hidden.register_backward_hook(handle_backward_hook)
        hidden_layer = F.relu(self.hidden(x))
        #hidden_layer.register_hook(handle_variable_hidden_hook)
        #self.hidden.weight.register_hook(handle_variable_predict_hook)

        #self.predict.register_forward_hook(handle_forward_hook)
        #self.predict.register_backward_hook(handle_backward_hook)
        direct_layer = self.predict(hidden_layer)
        #self.predict.weight.register_hook(handle_variable_predict_hook)
        #direct_layer.register_hook(handle_variable_predict_hook)

        return direct_layer

net = Net(n_features=1,n_hidden=3,n_output=1)
#net.register_forward_hook(handle_forward_hook)
#net.register_backward_hook(handle_backward_hook)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

for epoch in range(1):
    pre = net(x)
    #print('pre: ', pre)
    loss = criterion(pre,y)
    #print('loss: ',loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''
print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)
'''

torch.save(net.state_dict(),'FullyNN_params.pkl')  # save the original net

'''
test_data = Variable(torch.Tensor([[4.0]]))
net2 = torch.load('FullyNN.pkl')
print("Test: ",net2(test_data))
'''
'''
Net(
  (hidden): Linear(in_features=1, out_features=3, bias=True)
  (predict): Linear(in_features=3, out_features=1, bias=True)
)
hidden.weight 
 1.0860
-1.2707
 0.5386
[torch.FloatTensor of size 3x1]

hidden.bias 
-0.3009
-0.4270
 0.1734
[torch.FloatTensor of size 3]

predict.weight 
 0.9724  1.3310  0.3006
[torch.FloatTensor of size 1x3]

predict.bias 
1.00000e-02 *
  2.8504
[torch.FloatTensor of size 1]


***********forward_hook***************
Forward Input (Variable containing:
-1.0000
-0.5000
 0.0000
 0.5000
 1.0000
[torch.FloatTensor of size 5x1]
,)
Output Output Variable containing:
-0.4607  0.0833 -0.2314
-0.1608 -0.0196  0.0230
 0.1390 -0.1224  0.2774
 0.4389 -0.2253  0.5317
 0.7387 -0.3282  0.7861
[torch.FloatTensor of size 5x3]

**************************
***********forward_hook***************
Forward Input (Variable containing:
 0.0000  0.0833  0.0000
 0.0000  0.0000  0.0230
 0.1390  0.0000  0.2774
 0.4389  0.0000  0.5317
 0.7387  0.0000  0.7861
[torch.FloatTensor of size 5x3]
,)
Output Output Variable containing:
-0.0245
-0.0473
-0.1005
-0.1492
-0.1979
[torch.FloatTensor of size 5x1]

**************************


***********backward_hook***************
Linear(in_features=3, out_features=1, bias=True)
Grad Input (Variable containing:
-0.4704
-0.1413
-0.0725
-0.2185
-0.4815
[torch.FloatTensor of size 5x1]
, Variable containing:
-0.0134 -0.0992  0.1058
-0.0040 -0.0298  0.0318
-0.0021 -0.0153  0.0163
-0.0062 -0.0461  0.0492
-0.0137 -0.1015  0.1083
[torch.FloatTensor of size 5x3]
, Variable containing:
-0.4617
-0.0392
-0.5180
[torch.FloatTensor of size 3x1]
)
Grad Output (Variable containing:
-0.4704
-0.1413
-0.0725
-0.2185
-0.4815
[torch.FloatTensor of size 5x1]
,)
**************************
***********backward_hook***************
Linear(in_features=1, out_features=3, bias=True)
Grad Input (Variable containing:
 0.0000 -0.0992  0.0000
 0.0000  0.0000  0.0318
-0.0021  0.0000  0.0163
-0.0062  0.0000  0.0492
-0.0137  0.0000  0.1083
[torch.FloatTensor of size 5x3]
, None, Variable containing:
-0.0168  0.0992  0.1170
[torch.FloatTensor of size 1x3]
)
Grad Output (Variable containing:
 0.0000 -0.0992  0.0000
 0.0000  0.0000  0.0318
-0.0021  0.0000  0.0163
-0.0062  0.0000  0.0492
-0.0137  0.0000  0.1083
[torch.FloatTensor of size 5x3]
,)
**************************
'''
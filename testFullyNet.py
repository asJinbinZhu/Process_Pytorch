'''

1. Train a network
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(1)

def handle_variable_hidden_hook(grade):
    print('***********hidden_hook***************')
    grade.data[0][2] = 0.1000
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

def handle_variable_predict_hook(grade):
    print('***********predict_hook***************')
    print('grade: ',grade)
    # modify
    #grade.data[0] = 0
    print('**************************')

def handle_variable_weight_hidden_hook(grade):
    print('***********handle_variable_weight_hidden_hook***************')
    print('+++++++++++handle_variable_weight_hidden_hook+++++++++++++++')
    print(grade)
    #grade.data[0][2] = 0.1000
    #grade.data[0][0] = 0.0
    #grade.data[0][2] = 0.0
    #print('grade: ',grade.data[0][1])
    #grade.data[0] = 0
    print('**************************')


x = torch.unsqueeze(torch.linspace(-1,1,5),dim=1) # x data(Tensor), shape(100,1)
y= x.pow(2)+0.2*torch.rand(x.size())

x,y = Variable(x),Variable(y)
print(x)
print(y)

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden_1,n_hidden_2,n_output):
        super(Net,self).__init__()
        self.hidden_1 = torch.nn.Linear(n_features,n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1,n_hidden_2)
        self.predict = torch.nn.Linear(n_hidden_2,n_output)

    def forward(self, x):
        hidden_layer_1 = F.sigmoid(self.hidden_1(x))
        #print('hidden_layer_1 value: ', hidden_layer_1)  # output of hidden layer
        #hidden_layer_1.register_hook(handle_variable_hidden_hook)  # gradient of direct hidden's output
        #print("hidden_layer_1.weight ",self.hidden_1.weight)
        #self.hidden_1.weight.register_hook(handle_variable_weight_hidden_hook)

        hidden_layer_2 = F.sigmoid(self.hidden_2(hidden_layer_1))
        #print('hidden_layer_2 value: ', hidden_layer_2)  # output of hidden layer
        #hidden_layer_2.register_hook(handle_variable_hidden_hook)  # gradient of direct hidden's output
        #self.predict.weight.register_hook(handle_variable_weight_hidden_hook)

        direct_layer = self.predict(hidden_layer_2)
        #print('predict value: ', direct_layer)  # output of direct layer
        #direct_layer.register_hook(handle_variable_predict_hook)  # gradient of direct layer's output
        self.predict.weight.register_hook(handle_variable_weight_hidden_hook)

        return direct_layer

net = Net(n_features=1,n_hidden_1=3,n_hidden_2=4,n_output=1)

print('******************************Before training*****************')
print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

for epoch in range(1):
    pre = net(x)
    loss = criterion(pre,y)
    #print('loss: ',loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('******************************After training*****************')
print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)

torch.save(net.state_dict(),'FullyNN_params.pkl')  # save the original net

'''
test_data = Variable(torch.Tensor([[4.0]]))
net2 = torch.load('FullyNN.pkl')
print("Test: ",net2(test_data))
'''


'''
Variable containing:
-1.0000
-0.5000
 0.0000
 0.5000
 1.0000
[torch.FloatTensor of size 5x1]

Variable containing:
 1.1515
 0.3059
 0.0806
 0.3969
 1.0059
[torch.FloatTensor of size 5x1]

Net(
  (hidden_1): Linear(in_features=1, out_features=3, bias=True)
  (hidden_2): Linear(in_features=3, out_features=4, bias=True)
  (predict): Linear(in_features=4, out_features=1, bias=True)
)
hidden_1.weight 
 0.9810
-1.4401
 0.9790
[torch.FloatTensor of size 3x1]

hidden_1.bias 
-0.0178
-0.2813
-0.0695
[torch.FloatTensor of size 3]

hidden_2.weight 
 0.0285  0.2109 -0.2250
-0.0421 -0.0520  0.0837
 0.5698  1.1455  0.6604
-0.5084 -0.6535 -0.3866
[torch.FloatTensor of size 4x3]

hidden_2.bias 
-0.2490
-0.1850
-0.5467
 0.5498
[torch.FloatTensor of size 4]

predict.weight 
 0.2718 -0.4888  0.8185 -0.7855
[torch.FloatTensor of size 1x4]

predict.bias 
 0.5125
[torch.FloatTensor of size 1]
'''
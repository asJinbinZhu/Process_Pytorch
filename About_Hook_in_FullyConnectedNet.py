import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#torch.manual_seed(1)

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
    print('**************************')

def handle_variable_hook(grade):
    print('***********variable_hook***************')
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

x = torch.unsqueeze(torch.linspace(-1,1,5),dim=1) # x data(Tensor), shape(100,1)
y= x.pow(2)+0.2*torch.rand(x.size())

print('x=: ',x)
print('y=: ',y)

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
        print('hidden value: ',x)
        x = self.predict(x)
        print('predict value: ',x)
        x.register_hook(handle_variable_hook)
        return x
net = Net(n_features=1,n_hidden=3,n_output=1)
#net.register_forward_hook(handle_forward_hook)
#net.register_backward_hook(handle_backward_hook)
#net.hidden.weight.register_hook(handle_variable_hook)
#net.predict.weight.register_hook(handle_variable_hook)

print(net)
params = net.state_dict()
for k,v in params.items():
    print(k,v)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

for epoch in range(1):
    pre = net(x)
    loss = criterion(pre,y)
    print('loss: ',loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

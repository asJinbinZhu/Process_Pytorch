'''
2. read model
3. get gradient of neurals
4. update the gradient
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F

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

def handle_variable_hidden_hook(grade):
    print('***********hidden_hook***************')
    grade.data[0][2] = 0.1000
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

def handle_variable_predict_hook(grade):
    print('***********predict_hook***************')
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        hidden_layer = F.relu(self.hidden(x))
        print('hidden value: ', hidden_layer)  # output of hidden layer
        hidden_layer.register_hook(handle_variable_hidden_hook)  # gradient of direct hidden's output

        direct_layer = self.predict(hidden_layer)
        print('predict value: ', direct_layer)  # output of direct layer
        direct_layer.register_hook(handle_variable_predict_hook)  # gradient of direct layer's output

        return direct_layer

net = Net(n_features=1,n_hidden=3,n_output=1)
net.load_state_dict(torch.load('FullyNN_params.pkl'))


print(net)
# Got it
print('--------------------------------------Before--------------------------------------')
params = net.state_dict()
for k,v in params.items():
    print(k,v)
print('--------------------------------------End Before-----------------------------------')

# one time train to get the parameters
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
test_data = Variable(torch.Tensor([[0.6000]]))
target_data = Variable(torch.Tensor([[0.3659]]))
for epoch in range(1):
    pre = net(test_data)
    loss = criterion(pre,target_data)
    #print('loss: ',loss)
    #optimizer.zero_grad()
    loss.backward()
    #optimizer.step()

print('--------------------------------------After--------------------------------------')
params = net.state_dict()
for k,v in params.items():
    print(k,v)
print('--------------------------------------End After-----------------------------------')
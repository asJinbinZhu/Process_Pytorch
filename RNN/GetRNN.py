'''
2. read model
3. get gradient of neurals
4. update the gradient
'''
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
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
    # modify
    #grade.data[0] = 0
    print('**************************')

test_data = dsets.MNIST(
    root='../mnist/',
    train=False,
    transform=transforms.ToTensor(),
)
test_x = Variable(test_data.test_data,volatile=True).type(torch.FloatTensor)[:100]/255
test_y = Variable(test_data.test_labels,volatile=True).type(torch.FloatTensor)[:100]/255


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = torch.nn.Linear(64,10)

    def forward(self, x):
        r_out,(h_n,h_c)=self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
rnn.load_state_dict(torch.load('rnn.pkl'))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(),lr=0.01)

for epoch in range(1):
    test_output = rnn(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    loss = criterion(pred_y, test_y)
    # print('loss: ',loss)
    # optimizer.zero_grad()
    loss.backward()


'''
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
torch.save(net.state_dict(),'ApproxNN_params.pkl')  # save the approx_ed net

print('--------------------------------------After--------------------------------------')
params = net.state_dict()
for k,v in params.items():
    print(k,v)
print('--------------------------------------End After-----------------------------------')
'''
'''
1. inject err randomly
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random

def tool_injectErr(len):
    who = []
    threshod = 0.5
    ifInject = random.random()
    howMany = random.randint(1,len)
    if ifInject > threshod:
        who.append(howMany)
        for i in range(howMany):
            who.append(random.randint(0,len-1))
    return who

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        hidden_layer = F.relu(self.hidden(x))
        print('hidden value_ before injection: ', hidden_layer)  # output of hidden layer

        # inject
        # 1. judge whether inject this layer
        # 2. if true, cal how many(x) neurals to inject
        # 3. select x neurals randomly
        tool_injectErr(hidden_layer.data[0].size()[0])

        direct_layer = self.predict(hidden_layer)
        print('predict value_before injection: ', direct_layer)  # output of direct layer


        return direct_layer

net = Net(n_features=1,n_hidden=3,n_output=1)
net.load_state_dict(torch.load('ApproxNN_params.pkl'))

test_data = Variable(torch.Tensor([[0.6000]]))
print(net(test_data))




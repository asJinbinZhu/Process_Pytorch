import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)

LR = 0.01
BARCH_SIZE = 32
EPOCH = 12

# fake data
x = torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plt.plot(x.numpy(),y.numpy())
# plt.show()

torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=BARCH_SIZE,shuffle=True,num_workers=2)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(1,20)
        self.predict = torch.nn.Linear(20,1)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

criterion = torch.nn.MSELoss()
losses_history = [[],[],[],[]]

# training
for epoch in range(EPOCH):
    print('Epoch', epoch)
    for step, (batch_x,batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers,losses_history):
            out = net(b_x)
            loss = criterion(out,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data[0])

labels = ['SGD','Momentum','RMSprop','Adam']
for i,l_his in enumerate(losses_history):
    plt.plot(l_his,label=labels[i])
plt.legend(loc='best')
plt.ylabel('Step')
plt.xlabel('Loss')
plt.ylim((0,0.2))
plt.show()
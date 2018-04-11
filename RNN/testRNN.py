import torch
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

torch.manual_seed(1)
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='../mnist',
    train=True,
    transform=torchvision.transforms.ToTensor,
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i'%train_data.train_labels[0])
#plt.show()

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='../mnist',
    train=False,
    transform=torchvision.transforms.ToTensor,
)
test_x = Variable(test_data.test_data,volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels.numpy().squeeze()[:2000]

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = torch.nn.Linear(64,10)

    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out
rnn = RNN()
print(rnn)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)

train_data = enumerate(train_data)

'''
# training
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28))
        b_y = Variable(y)

        output = rnn(b_x)
        loss = criterion(output,b_y)

        optimizer.zero_grad()
        loss.bakcward()
        optimizer.step()
'''
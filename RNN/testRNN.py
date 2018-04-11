import torch
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data

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
    download=DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(
    root='../mnist',
    train=False
)

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]

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
        r_out,(h_n,h_c) = self.rnn(x)
        out = self.out(r_out[:,-1,:])
        return out
rnn = RNN()
print(rnn)
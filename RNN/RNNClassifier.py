import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as dsets

BATCH_SIZE=64
DOWNLOAD=True
LR = 0.02


train_data = dsets.MNIST(
    root='../mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD,
)
# show one picture
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i'%train_data.train_labels[0])
#plt.show()

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data = dsets.MNIST(
    root='../mnist/',
    train=False,
    transform=transforms.ToTensor(),
)
test_x = Variable(test_data.test_data,volatile=True).type(torch.FloatTensor)[:100]/255
test_y = test_data.test_labels.numpy().squeeze()[:100]

def handle_variable_hidden_hook(grade):
    print('***********hidden_hook***************')
    grade.data[0][0] = 0.0
    grade.data[11][1] = 0.0
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

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
        r_out,(h_n,h_c)=self.rnn(x,None)
        r_out.register_hook(handle_variable_hidden_hook)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
print(rnn)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)

for epoch in range(1):
    for steps,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28))
        b_y = Variable(y)

        out_put = rnn(b_x)
        loss = criterion(out_put,b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size)
            print('Epoch: ',epoch,' |loss: %.4f'%loss.data[0],' |accuracy: %.2f'%accuracy)

test_output = rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10],'real numbers')

torch.save(rnn.state_dict(),'rnn.pkl')
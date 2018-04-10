import torch
from torch.autograd import Variable
import torch.nn.functional as F

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
y_data = Variable(torch.Tensor([[0.],[0.],[1.],[1.]]))

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pre = F.sigmoid(self.linear(x))
        return y_pre

net = LogisticRegression()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)

# training
for epoch in range(100):
    y_pre = net(x_data)

    loss = criterion(y_pre,y_data)
    print('Loss: ',loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_data = Variable(torch.Tensor([1.0]))
print('Prediction: ',net(test_data))
import torch
from torch.autograd import Variable
from LinearRegressionFromMoFan import Net
from utils.Visualizer import make_dot

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) # x data(Tensor), shape(100,1)
y= x.pow(2)+0.2*torch.rand(x.size())
x,y = Variable(x),Variable(y)

net = Net(n_features=1,n_hidden=10,n_output=1)
prediction = net(x)

g = make_dot(prediction)
g.view()


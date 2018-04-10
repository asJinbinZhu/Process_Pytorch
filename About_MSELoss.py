import torch
from torch.autograd import Variable

loss_fun = torch.nn.MSELoss()
input = Variable(torch.randn(1))
target = Variable(torch.randn(1))

loss = loss_fun(input,target)
print(input)
print(target)
print(loss)

'''
Variable containing:
 0.0482  1.7498  0.9358
[torch.FloatTensor of size 1x3]

Variable containing:
 0.4471  2.1832 -1.1156
[torch.FloatTensor of size 1x3]

Variable containing:
 1.5184
[torch.FloatTensor of size 1]
'''
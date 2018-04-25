import torch

torch.manual_seed(1)

a = torch.randn(2, 3)
# print(a)
'''
 0.6614  0.2669  0.0617
 0.6213 -0.4519 -0.1661
[torch.FloatTensor of size 2x3]
'''
b = a.view(1, -1)
# print(b)
'''
 0.6614  0.2669  0.0617  0.6213 -0.4519 -0.1661
[torch.FloatTensor of size 1x6]
'''
b = a.view(2, 1, -1)
# print(b)
# print('*'*10)
b = b.view(2, -1)
# print(b)
'''
(0 ,.,.) = 
  0.6614  0.2669  0.0617  0.6213 -0.4519 -0.1661
[torch.FloatTensor of size 1x1x6]
'''

a = torch.randn(2, 3)
print(a)
print(a.squeeze(0))

a = torch.randn(1,3)
print(a)
print(a.squeeze(0))

a = torch.randn(1,1,3)
print(a)
print(a.squeeze(0))
b = a.squeeze(0)
print(b.unsqueeze(0))
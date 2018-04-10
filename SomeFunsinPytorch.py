import torch

a = torch.Tensor(2,3)
print(a)
'''
-1.1143e+20  4.5741e-41  6.8438e-38
 0.0000e+00  4.4842e-44  0.0000e+00
[torch.FloatTensor of size 2x3]
'''
b = a.view(1,-1)    # multi rows to one row
print(b)
'''
-1.1143e+20  4.5741e-41  6.8438e-38  0.0000e+00  4.4842e-44  0.0000e+00
[torch.FloatTensor of size 1x6]
'''

b = torch.Tensor(1,3)
print(b)
print(b.squeeze(0))
'''
1.00000e-38 *
  6.3424
  0.0000
  6.3417
[torch.FloatTensor of size 3]
'''

a = torch.Tensor(3)
print(a)
'''
 7.5229e+00
 4.5715e-41
 3.4292e-38
[torch.FloatTensor of size 3]
'''
print(a.unsqueeze(0))
'''
1.00000e-38 *
  4.6826  0.0000  4.4132
[torch.FloatTensor of size 1x3]
'''





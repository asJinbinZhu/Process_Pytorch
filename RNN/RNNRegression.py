import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

steps = np.linspace(0,np.pi*2,100,dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps,x_np,'r-',label='target(cos)')
plt.plot(steps,y_np,'b-',label='input(sin)')
plt.legend(loc='best')
plt.show()

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out=torch.nn.Linear(32,1)
    def forward(self, x,h_state):
        r_out,h_state = self.rnn(x,h_state)
        out = []

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Sequence(torch.nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = torch.nn.LSTM(1,64)
        self.lstm2 = torch.nn.LSTM(64,1)

    def forward(self, seq, hc=None):
        if hc == None:
            hc1,hc2 = None,None
        else:
            hc1,hc2 = hc

        lstm1_out,hc1 = self.lstm1(seq,hc1)
        lstm2_out,hc2 = self.lstm2(seq,hc2)
        return torch.stack(out).squeeze(1),(hc1,hc2)
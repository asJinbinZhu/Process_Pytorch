import torch
from torch.autograd import Variable

word_to_ix = {'hello':0,'world':1}
embeds = torch.nn.Embedding(2,5)
hello_idx = embeds(Variable(torch.LongTensor([word_to_ix['hello']])))
print(hello_idx)

import torch
from torch.autograd import Variable

time_steps = 3
batch_size = 2
in_size = 2
classes_NO = 5
num_layers = 2

model = torch.nn.LSTM(in_size, classes_NO, num_layers=num_layers)
print(model)

inputs = Variable(torch.randn(time_steps, batch_size, in_size))
outputs, _ = model(inputs)
last_output = outputs[-1]
print(last_output)

criterion = torch.nn.CrossEntropyLoss()
target = Variable(torch.LongTensor(batch_size).random_(0, classes_NO-1))
loss = criterion(last_output, target)
loss.backward()
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

def handle_forward_hook(module, input, output):
    print('***********forward_hook***************')
    print(module)
    print('Forward Input', input)
    print('Output Output', output)
    print('**************************')


def handle_backward_hook(module, input, output):
    print('***********backward_hook***************')
    print(module)
    print('Grad Input', input)
    print('Grad Output', output)
    print('**************************')


def handle_variable_hidden_hook(grade):
    print('***********hidden_hook***************')
    # grade.data[0][0] = 0.0
    # grade.data[0][1] = 0.0
    print('grade: ', grade)
    # grade.data[0] = 0
    print('**************************')


def handle_variable_predict_hook(grade):
    print('***********predict_hook***************')
    print('grade: ', grade)
    # modify
    # grade.data[0] = 0
    print('**************************')

time_steps = 1
batch_size = 2
in_size = 2
classes_NO = 3
num_layers = 2

model = torch.nn.LSTM(in_size, classes_NO, num_layers=num_layers)
model.register_forward_hook(handle_forward_hook)
print(model)
params = model.state_dict()
for k,v in params.items():
    print(k,v)


input_seq = Variable(torch.randn(time_steps, batch_size, in_size))

# initialize the hidden state.
hidden = (Variable(torch.zeros(num_layers, 2, classes_NO)),
          Variable(torch.zeros((num_layers, 2, classes_NO))))

target = Variable(torch.LongTensor(batch_size).random_(0, classes_NO-1))

'''
def decoder(inputs, lstm, hidden):
    hs = []
    for input in inputs:
        out, (h, c) = lstm(input.view(1, len(input), -1), hidden)
        h.register_hook(handle_variable_predict_hook)
        hs.append(h)
        hidden = (h, c)
        lstm_out = torch.stack(hs, dim=0)
    return lstm_out

output_seq = decoder(input_seq, model, hidden)
last_output = output_seq[-1][1]
print('last_output_decoder: ', last_output)
'''

output_seq, _ = model(input_seq)
last_output = output_seq[-1]
#print('last_output_no_decoder', last_output)


criterion = torch.nn.CrossEntropyLoss()

loss = criterion(last_output, target)
loss.backward()
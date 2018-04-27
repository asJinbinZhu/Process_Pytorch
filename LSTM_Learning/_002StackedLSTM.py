import torch
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(1)


def handle_forward_hook(module,input,output):
    print('***********forward_hook***************')
    print(module)
    print('Forward Input', input)
    print('Output Output', output)
    print('**************************')

def handle_backward_hook(module,input,output):
    print('***********backward_hook***************')
    print(module)
    print('Grad Input',input)
    print('Grad Output',output)
    print('**************************')

def handle_variable_hook(grade):
    print('***********hidden_hook***************')
    #grade.data[0] = 0.0
    print('grade: ',grade)
    #grade.data[0] = 0
    #print('**************************')

# Hyper parameters
nLayers = 3
input_size = 2
hidden_size = 3
time_steps = 1
batch_size = 2

# training data
input = Variable(torch.rand(time_steps, batch_size, input_size))
target = Variable(torch.LongTensor(batch_size).random_(0, hidden_size-1))

# stacked lstm model
model = torch.nn.ModuleList()
for i in range(nLayers):
    input_size = input_size if i == 0 else hidden_size
    model.append(torch.nn.LSTM(input_size, hidden_size))

print(model)
'''
params = model.state_dict()
for k,v in params.items():
    print(k,v)
'''

# forward
outputs = []
for i in range(nLayers):
    #if i != 0:
        #input = F.dropout(input, p=0.2, training=True)
    output, (h, c) = model[i](input)
    if i == 0:
        c[0,0,0]=0.3
    output.register_hook(handle_variable_hook)

    outputs.append(output)

    input = output
    hidden = (h, c)

last_output = outputs[-1]
print('last_output', last_output)

# training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# backward
loss = criterion(last_output.squeeze(0), target)
loss.backward()
optimizer.step()

'''
params = model.state_dict()
for k,v in params.items():
    print(k,v)
'''
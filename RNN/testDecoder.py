import torch
import torch.optim as optim
import torch.nn as nn
import time
from torch.autograd import Variable
from time import sleep

# torch.backends.cudnn.enabled=False
# torch.cuda.set_device(1)

input_size = 2
hidden_size = 3
batch_size = 1
time_steps = 1
num_layers = 2
lr = 4e-2


def handle_backward_hook(module,input,output):
    print('***********backward_hook***************')
    print(module)
    print('Grad Input',input)
    print('Grad Output',output)
    print('**************************')

def handle_variable_hidden_hook(grade):
    print('***********hidden_hook***************')
    #grade.data[0][0] = 0.0
    #grade.data[0][1] = 0.0
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

def initHidden(bsz):
    return (Variable(torch.FloatTensor(num_layers, bsz, hidden_size).zero_()),
            Variable(torch.FloatTensor(num_layers, bsz, hidden_size).zero_()))


def resetHidden(hidden):
    return (Variable(hidden[0].data.zero_()), Variable(hidden[1].data.zero_()))


model = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
#model.register_backward_hook(handle_backward_hook)

input = torch.randn(time_steps, batch_size, input_size)
target = torch.randn(time_steps, batch_size, hidden_size)

optimizer = optim.SGD(model.parameters(),
                      lr=lr,
                      momentum=0.9,
                      dampening=0.0,
                      weight_decay=0.0
                      )

criterion = nn.MSELoss()

loss = 0

hidden = initHidden(batch_size)
input = Variable(input)
target = Variable(target, requires_grad=False)

for i in range(1):
    start = time.time()
    for epoch in range(1):
        print(epoch)
        loss = 0
        model.zero_grad()
        optimizer.zero_grad()

        hidden = initHidden(batch_size)
        #        output, hidden = model(input, hidden)
        outputs = []
        for j in range(input.size(0)):
            output, hidden = model(input[j].view(1, *input[j].size()), hidden)
            output.register_hook(handle_variable_hidden_hook)
            outputs.append(output)
        outputs = torch.cat(outputs, 0)
        output = outputs

        loss = criterion(output, target)
        loss.backward(retain_variables=True)

        optimizer.step()

    print("Test ran in " + str(time.time() - start) + " seconds")
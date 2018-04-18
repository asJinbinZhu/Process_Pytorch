import torch
from torch.autograd import Variable

torch.manual_seed(1)

time_steps = 2
batch_size = 1
in_size = 2
classes_no = 3

input_seq = [Variable(torch.randn(time_steps,batch_size,in_size))]
target = Variable(torch.LongTensor(batch_size).random_(0,classes_no-1))

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

def handle_variable_hidden_hook(grade):
    print('***********hidden_hook***************')
    grade.data[0][0] = 0.0
    grade.data[11][1] = 0.0
    print('grade: ',grade)
    #grade.data[0] = 0
    print('**************************')

def handle_variable_predict_hook(grade):
    print('***********predict_hook***************')
    print('grade: ',grade)
    # modify
    #grade.data[0] = 0
    print('**************************')

class LSTMTagger(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_nums):
        super(LSTMTagger, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_nums = layer_nums

        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_nums)

    def init_hidden(self):
        return (Variable(torch.randn(self.layer_nums, batch_size, classes_no), requires_grad=True),
                Variable(torch.randn((self.layer_nums,batch_size,classes_no)), requires_grad=True))

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        #out.register_hook(handle_variable_hidden_hook)  # gradient of direct out # get out's grade
        #print(hidden[1][0][0])
        hidden[0][0][0][0].register_hook(handle_backward_hook)
        hidden[1][0][0][0].register_hook(handle_variable_hidden_hook)
        #self.lstm.weight_ih_l0.register_hook(handle_variable_hidden_hook) # set grade to 0.0
        last_out = out[-1]
        return last_out

model = LSTMTagger(in_size, classes_no, 1)
'''
print(model)
params = model.state_dict()
for k,v in params.items():
    print(k, v)
'''

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1):
    hidden = model.init_hidden()
    for ipt in input_seq:
        out = model(ipt, hidden)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''
for epoch in range(1):
    out = model(input_seq,hidden)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''

'''
params = model.state_dict()
for k,v in params.items():
    print(k, v)
'''
torch.save(model.state_dict(),'SimpleLSTM_params.pkl')  # save the original net
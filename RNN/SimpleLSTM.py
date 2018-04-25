import torch
from torch.autograd import Variable

torch.manual_seed(1)

time_steps = 1
batch_size = 1
in_size = 2
classes_no = 3

input_seq = [Variable(torch.randn(time_steps,batch_size,in_size))]
target = Variable(torch.LongTensor(batch_size).random_(0,classes_no-1))
print('input: ', input_seq, 'output: ', target)

def handle_forward_hook(module,input,output):
    print('***********forward_hook***************')
    #print(module)
    print('Forward Input', input)
    print('Output Output', output)    #output[0] - out; output[1][0] - h; output[1][1] - c
    print('**************************')

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
        h0 = Variable(torch.randn(self.layer_nums, batch_size, classes_no), requires_grad=True)
        c0 = Variable(torch.randn(self.layer_nums, batch_size, classes_no), requires_grad=True)
        return h0, c0

    def forward(self, x):
        h, c = self.init_hidden()
        #self.lstm.register_forward_hook(handle_forward_hook)
        #self.lstm.register_backward_hook(handle_backward_hook)
        out, (h, c) = self.lstm(x, (h, c))

        #print('OUT: ', out)
        #out.register_hook(handle_variable_hidden_hook)
        # out.register_hook(handle_variable_hidden_hook)
        last_out = out[-1]
        return last_out

model = LSTMTagger(in_size, classes_no, 1)
#model.register_backward_hook(handle_backward_hook)

print(model)
params = model.state_dict()
for k,v in params.items():
    print(k,v)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1):
    for ipt in input_seq:
        out = model(ipt)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''
input:  [Variable containing:
(0 ,.,.) = 
  0.6614  0.2669
[torch.FloatTensor of size 1x1x2]

LSTMTagger(
  (lstm): LSTM(2, 3)
)
lstm.weight_ih_l0 
 0.3462 -0.1188
 0.2937  0.0803
-0.0707  0.1601
 0.0285  0.2109
-0.2250 -0.0421
-0.0520  0.0837
-0.0023  0.5047
 0.1797 -0.2150
-0.3487 -0.0968
-0.2490 -0.1850
 0.0276  0.3442
 0.3138 -0.5644
[torch.FloatTensor of size 12x2]

lstm.weight_hh_l0 
 0.3579  0.1613  0.5476
 0.3811 -0.5260 -0.5489
-0.2785  0.5070 -0.0962
 0.2471 -0.2683  0.5665
-0.2443  0.4330  0.0068
-0.3042  0.2968 -0.3065
 0.1698 -0.1667 -0.0633
-0.5551 -0.2753  0.3133
-0.1403  0.5751  0.4628
-0.0270 -0.3854  0.3516
 0.1792 -0.3732  0.3750
 0.3505  0.5120 -0.3236
[torch.FloatTensor of size 12x3]

lstm.bias_ih_l0 
-0.0950
-0.0112
 0.0843
-0.4382
-0.4097
 0.3141
-0.1354
 0.2820
 0.0329
 0.1896
 0.1270
 0.2099
[torch.FloatTensor of size 12]

lstm.bias_hh_l0 
 0.2862
-0.5347
 0.2906
-0.4059
-0.4356
 0.0351
-0.0984
 0.3391
-0.3344
-0.5133
 0.4202
-0.0856
[torch.FloatTensor of size 12]
'''

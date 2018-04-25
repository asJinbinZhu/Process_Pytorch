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
        print('h: ', h, 'c: ', c)
        #self.lstm.register_forward_hook(handle_forward_hook)
        #self.lstm.register_backward_hook(handle_backward_hook)
        out, (h, c) = self.lstm(x, (h, c))

        #print('OUT: ', out, 'h: ', h, 'c: ', c)
        #out.register_hook(handle_variable_hidden_hook)
        # out.register_hook(handle_variable_hidden_hook)
        last_out = out[-1]
        return last_out

model = LSTMTagger(in_size, classes_no, 3)
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

h:  Variable containing:
(0 ,.,.) = 
  0.0000  0.0316  0.4501
[torch.FloatTensor of size 1x1x3]
 c:  Variable containing:
(0 ,.,.) = 
 -0.1132 -0.3688 -1.7254
[torch.FloatTensor of size 1x1x3]


***********forward_hook***************
Forward Input (Variable containing:
(0 ,.,.) = 
  0.6614  0.2669
[torch.FloatTensor of size 1x1x2]
, (Variable containing:
(0 ,.,.) = 
  0.0000  0.0316  0.4501
[torch.FloatTensor of size 1x1x3]
, Variable containing:
(0 ,.,.) = 
 -0.1132 -0.3688 -1.7254
[torch.FloatTensor of size 1x1x3]
))
Output Output (Variable containing:
(0 ,.,.) = 
 -0.0520  0.0958 -0.4176
[torch.FloatTensor of size 1x1x3]
, (Variable containing:
(0 ,.,.) = 
 -0.0520  0.0958 -0.4176
[torch.FloatTensor of size 1x1x3]
, Variable containing:
(0 ,.,.) = 
 -0.1296  0.1391 -1.1395
[torch.FloatTensor of size 1x1x3]
))
**************************

***********backward_hook***************
LSTM(2, 3)
Grad Input (Variable containing:
(0 ,.,.) = 
  0.3505 -0.5937  0.2432
[torch.FloatTensor of size 1x1x3]
,)
Grad Output (Variable containing:
(0 ,.,.) = 
  0.3505 -0.5937  0.2432
[torch.FloatTensor of size 1x1x3]
,)
**************************

OUT:  Variable containing:
(0 ,.,.) = 
 -0.0520  0.0958 -0.4176
[torch.FloatTensor of size 1x1x3]
 h:  Variable containing:
(0 ,.,.) = 
 -0.0520  0.0958 -0.4176
[torch.FloatTensor of size 1x1x3]
 c:  Variable containing:
(0 ,.,.) = 
 -0.1296  0.1391 -1.1395
[torch.FloatTensor of size 1x1x3]
'''
#***************************************num_layers = 2********************************************
'''
LSTMTagger(
  (lstm): LSTM(2, 3, num_layers=2)
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

lstm.weight_ih_l1 
 0.3247  0.1856 -0.4329
 0.1160  0.1387 -0.3866
-0.2739  0.1969  0.1034
-0.2456 -0.1748  0.5288
-0.1068  0.3255  0.2500
-0.3732 -0.4910  0.5542
 0.0301  0.3957  0.1196
 0.1857  0.4313  0.5475
-0.3831  0.0722  0.4309
 0.4183  0.3587 -0.4178
-0.4158 -0.3492  0.0725
 0.5754 -0.3647  0.3077
[torch.FloatTensor of size 12x3]

lstm.weight_hh_l1 
-0.3196 -0.5428 -0.1227
 0.3327  0.5360 -0.3586
 0.1253  0.4982  0.3826
 0.3598  0.4103  0.3652
 0.1491 -0.3948 -0.4848
-0.2646 -0.0672 -0.3539
 0.2112  0.1787 -0.1307
 0.2219  0.1866  0.3525
 0.3888 -0.1955  0.5641
-0.0667 -0.0198 -0.5449
-0.3716 -0.3373 -0.2469
 0.4105 -0.1887 -0.4314
[torch.FloatTensor of size 12x3]

lstm.bias_ih_l1 
 0.2221
 0.1848
 0.3739
-0.2988
 0.1252
-0.2102
-0.1297
-0.4601
-0.2631
-0.1768
 0.2469
 0.1055
[torch.FloatTensor of size 12]

lstm.bias_hh_l1 
 0.1426
 0.5763
 0.5627
 0.3938
 0.0184
-0.3994
 0.4512
-0.1444
-0.0467
-0.4974
-0.1140
-0.3724
[torch.FloatTensor of size 12]

input:  [Variable containing:
(0 ,.,.) = 
  0.6614  0.2669
[torch.FloatTensor of size 1x1x2]
] output:  Variable containing:
 1
[torch.LongTensor of size 1]

***********forward_hook***************
Forward Input (Variable containing:
(0 ,.,.) = 
  0.6614  0.2669
[torch.FloatTensor of size 1x1x2]
, (Variable containing:
(0 ,.,.) = 
 -0.9315 -1.8460  0.5781

(1 ,.,.) = 
 -0.9483  0.0943  0.6352
[torch.FloatTensor of size 2x1x3]
, Variable containing:
(0 ,.,.) = 
 -0.6054 -0.2838  1.0525

(1 ,.,.) = 
  0.2865  1.4871  1.3805
[torch.FloatTensor of size 2x1x3]
))
Output Output (Variable containing:
(0 ,.,.) = 
  0.0716  0.2260  0.0670
[torch.FloatTensor of size 1x1x3]
, (Variable containing:
(0 ,.,.) = 
 -0.1577  0.3191  0.0333

(1 ,.,.) = 
  0.0716  0.2260  0.0670
[torch.FloatTensor of size 2x1x3]
, Variable containing:
(0 ,.,.) = 
 -0.2696  0.4214  0.1538

(1 ,.,.) = 
  0.2575  0.4253  0.2852
[torch.FloatTensor of size 2x1x3]
))
**************************
OUT:  Variable containing:
(0 ,.,.) = 
  0.0716  0.2260  0.0670
[torch.FloatTensor of size 1x1x3]
 h:  Variable containing:
(0 ,.,.) = 
 -0.1577  0.3191  0.0333

(1 ,.,.) = 
  0.0716  0.2260  0.0670
[torch.FloatTensor of size 2x1x3]
 c:  Variable containing:
(0 ,.,.) = 
 -0.2696  0.4214  0.1538

(1 ,.,.) = 
  0.2575  0.4253  0.2852
[torch.FloatTensor of size 2x1x3]


h:  Variable containing:
(0 ,.,.) = 
 -0.9315 -1.8460  0.5781

(1 ,.,.) = 
 -0.9483  0.0943  0.6352
[torch.FloatTensor of size 2x1x3]
 c:  Variable containing:
(0 ,.,.) = 
 -0.6054 -0.2838  1.0525

(1 ,.,.) = 
  0.2865  1.4871  1.3805
[torch.FloatTensor of size 2x1x3]


input:  [Variable containing:
(0 ,.,.) = 
  0.6614  0.2669
[torch.FloatTensor of size 1x1x2]
] output:  Variable containing:
 1
[torch.LongTensor of size 1]

LSTMTagger(
  (lstm): LSTM(2, 3, num_layers=3)
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

lstm.weight_ih_l1 
 0.3247  0.1856 -0.4329
 0.1160  0.1387 -0.3866
-0.2739  0.1969  0.1034
-0.2456 -0.1748  0.5288
-0.1068  0.3255  0.2500
-0.3732 -0.4910  0.5542
 0.0301  0.3957  0.1196
 0.1857  0.4313  0.5475
-0.3831  0.0722  0.4309
 0.4183  0.3587 -0.4178
-0.4158 -0.3492  0.0725
 0.5754 -0.3647  0.3077
[torch.FloatTensor of size 12x3]

lstm.weight_hh_l1 
-0.3196 -0.5428 -0.1227
 0.3327  0.5360 -0.3586
 0.1253  0.4982  0.3826
 0.3598  0.4103  0.3652
 0.1491 -0.3948 -0.4848
-0.2646 -0.0672 -0.3539
 0.2112  0.1787 -0.1307
 0.2219  0.1866  0.3525
 0.3888 -0.1955  0.5641
-0.0667 -0.0198 -0.5449
-0.3716 -0.3373 -0.2469
 0.4105 -0.1887 -0.4314
[torch.FloatTensor of size 12x3]

lstm.bias_ih_l1 
 0.2221
 0.1848
 0.3739
-0.2988
 0.1252
-0.2102
-0.1297
-0.4601
-0.2631
-0.1768
 0.2469
 0.1055
[torch.FloatTensor of size 12]

lstm.bias_hh_l1 
 0.1426
 0.5763
 0.5627
 0.3938
 0.0184
-0.3994
 0.4512
-0.1444
-0.0467
-0.4974
-0.1140
-0.3724
[torch.FloatTensor of size 12]

lstm.weight_ih_l2 
 0.5305 -0.4991 -0.4500
-0.0196 -0.3122  0.2066
-0.2222 -0.2712  0.0327
 0.4179 -0.4061  0.2711
 0.3709  0.5648 -0.4041
 0.1398 -0.4269  0.4929
-0.2240  0.3478  0.0172
-0.0450 -0.0184  0.0981
 0.2722  0.0926  0.1761
-0.5193  0.4206  0.5034
 0.4772  0.4268 -0.4166
-0.2140  0.5091 -0.4397
[torch.FloatTensor of size 12x3]

lstm.weight_hh_l2 
 0.5238 -0.4541 -0.4067
 0.2823 -0.4148 -0.1323
 0.4200  0.4573  0.5460
-0.1172 -0.4488  0.5685
-0.1230 -0.2375  0.1407
-0.4038  0.3795  0.3618
-0.4581 -0.4742 -0.0506
 0.2425 -0.0167 -0.2928
 0.0132 -0.5427 -0.4080
-0.3843  0.4755  0.5089
-0.1961  0.0259  0.2575
 0.0692 -0.2891  0.3330
[torch.FloatTensor of size 12x3]

lstm.bias_ih_l2 
 0.3549
-0.0335
-0.0711
 0.5247
 0.5047
-0.3273
 0.5649
 0.1429
-0.3835
 0.3160
-0.4310
 0.5334
[torch.FloatTensor of size 12]

lstm.bias_hh_l2 
-0.3712
 0.1633
 0.1758
 0.1373
 0.4789
-0.2398
-0.2437
-0.5003
-0.0237
-0.2735
 0.0231
-0.1184
[torch.FloatTensor of size 12]

h:  Variable containing:
(0 ,.,.) = 
 -0.2939  0.5579  0.9284

(1 ,.,.) = 
  0.2211  0.3865 -1.0245

(2 ,.,.) = 
  0.0842  1.6088  1.6084
[torch.FloatTensor of size 3x1x3]
 c:  Variable containing:
(0 ,.,.) = 
  0.5800  1.4098 -1.9873

(1 ,.,.) = 
 -1.1417  0.1935 -2.8692

(2 ,.,.) = 
 -1.5053 -0.4115  0.0479
[torch.FloatTensor of size 3x1x3]


'''
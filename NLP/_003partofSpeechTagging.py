import torch
from torch.autograd import Variable
import torch.nn.functional as F


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


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read the book".split(), ["NN", "V", "DET", "NN"])
]

word_to_idx = {}
tag_to_idx = {}
for context, tag in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)

alphabet = "abcdefghijklmnopqrstuvwxyz"
char_to_idex = {}
for i in range(len(alphabet)):
    char_to_idex[alphabet[i]] = i

class CharLSTM(torch.nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = torch.nn.Embedding(n_char, char_dim)
        self.char_lstm = torch.nn.LSTM(char_dim, char_hidden, batch_first=True)
    def forward(self, x):
        x = self.char_embedding(x)
        _, h = self.char_lstm(x)
        return h[1]

class LSTMTagger(torch.nn.Module):
    def __init__(self, n_word, n_char, word_dim, char_dim, word_hidden, char_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_emebdding = torch.nn.Embedding(n_word, word_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        #self.char_lstm.register_backward_hook(handle_backward_hook)

        self.word_lstm = torch.nn.LSTM(word_dim+char_hidden, word_hidden, batch_first=True)
        #self.word_lstm.register_backward_hook(handle_backward_hook)

        self.linear1 = torch.nn.Linear(word_hidden, n_tag)
    def forward(self, x, word):
        char = torch.FloatTensor()
        for each in word:
            char_list = []
            for letter in each:
                char_list.append(char_to_idex[letter.lower()])
            char_list = torch.LongTensor(char_list)
            char_list = char_list.unsqueeze(0)
            '''
            if torch.cuda.is_available():
                tempchar = self.char_lstm(Variable(char_list).cuda())
            else:
                tempchar = self.char_lstm(Variable(char_list).cuda())
            '''
            tempchar = self.char_lstm(Variable(char_list))
            tempchar = tempchar.squeeze(0)
            char = torch.cat((char, tempchar.cpu().data), 0)
        '''
        if torch.cuda.is_available():
            char = char.is_cuda()
        '''
        char = Variable(char)
        x = self.word_emebdding(x)
        x = torch.cat((x, char), 1)
        x = x.unsqueeze(0)
        x, _ = self.word_lstm(x)
        x = x.squeeze(0)
        x = self.linear1(x)
        y = F.log_softmax(x, dim=1)
        return y

model = LSTMTagger(
    len(word_to_idx),
    len(char_to_idex),
    100,
    10,
    128,
    50,
    len(tag_to_idx)
)
'''
if torch.cuda.is_available():
    model = model.cuda()
'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

def make_sentence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx

for epoch in range(1):
    #print('*'*10)
    #print('epoch {}'.format(epoch + 1))
    running_loss = 0

    for data in training_data:
        word, tag = data
        #print('word: ', word)
        word_list = make_sentence(word, word_to_idx)
        tag = make_sentence(tag, tag_to_idx)
        '''
        if torch.cuda.is_available():
            word_list = word_list.cuda()
            tag = tag.cuda()
        '''
        # forward
        out = model(word_list, word)
        loss = criterion(out, tag)
        running_loss += loss.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #print('Loss: {}'.format(running_loss/len(data)))

#print()
input = make_sentence("Everybody ate the apple".split(), word_to_idx)
'''
if torch.cuda.is_available():
    input = input.cuda()
'''
out = model(input, "Tom ate an apple".split())
#print(out)
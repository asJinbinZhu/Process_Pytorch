import torch
from torch.autograd import Variable
import torch.nn.functional as F

import random
import DataLoader

EMBEDDING_DIM = 50
HIDDEN_DIM = 50
EPOCH = 1
best_dev_acc = 0.0

class LSTMClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = torch.nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            (Variable(torch.randn(1, 1, self.hidden_dim))),
            (Variable(torch.randn(1, 1, self.hidden_dim)))
        )

    def forward(self, input):
        embeds = self.word_embedding(input)
        x = embeds.view(len(input), 1, -1)
        out, hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(out[-1])
        y = F.log_softmax(y, dim=1)
        return y

def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right/len(truth)

train_data, dev_data, test_data, word_to_idx, label_to_idx = DataLoader.load_MR_data()
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(label_to_idx))

print(model)
'''
params = model.state_dict()
for k,v in params.items():
    print(k,v)
'''

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1):
    random.shuffle(train_data)
    for sent, label in train_data:
        # training data
        sent = DataLoader.prepare_sentence(sent, word_to_idx)
        label = DataLoader.prepare_label(label, label_to_idx)

        # forward
        pred = model(sent)
        loss = criterion(pred, label)
        #print('Epoch: ', epoch, 'Loss: ', loss.data[0])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''
def train():
    train_data, dev_data, test_data, word_to_idx, label_to_idx = DataLoader.load_MR_data()
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(label_to_idx))

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    no_up = 0
    for epoch in range(EPOCH):
        random.shuffle(train_data)
        print('epoch: %d start!'%epoch)

        train_epoch(model, train_data, criterion, optimizer, word_to_idx, label_to_idx, epoch)
        print('now, best dev acc: ', best_dev_acc)

def train_epoch(model, train_data, criterion, optimizer, word_to_idx, label_to_idx, epoch):
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:
        truth_res.append(label_to_idx[label])
        model.hidden = model.init_hidden()

        #training data
        sent = DataLoader.prepare_sentence(sent, word_to_idx)
        label = DataLoader.prepare_label(label, label_to_idx)

        #forward
        pred = model(sent)
        loss = criterion(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss : %g'%(epoch, count, loss.data[0]))

        pred_label = pred.data.max(1)[1].numpy()
        print(pred_label)
        pred_res.append(pred_label)


        optimizer.zero_grad()
        loss.backward()
    avg_loss /= len(train_data)
    #print('epoch: %d done! \n train avg_loss:%g , acc:%g' % (epoch, avg_loss, get_accuracy(truth_res, pred_res)))
train()
'''
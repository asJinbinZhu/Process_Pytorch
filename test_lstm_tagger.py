import torch
from torch.autograd import Variable
import torch.nn.functional as F

# training data
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read the book".split(), ["NN", "V", "DET", "NN"])
]

# index
word_to_idx = {}
tag_to_idx = {}
for content, tag in training_data:
    for word in content:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)
class LSTMTagger(torch.nn.Module):
    def __init__(self, word_num, word_dim, hidden_dim, tag_num):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(word_num, word_dim)
        self.lstm = torch.nn.LSTM(word_dim, hidden_dim)
        self.hidden2tag = torch.nn.Linear(hidden_dim, tag_num)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        hidden = (
            (Variable(torch.randn(1, 1, self.hidden_dim))),
            (Variable(torch.randn(1, 1, self.hidden_dim)))
        )
        return hidden
    def forward(self, sentence):
        embds = self.embedding(sentence)
        out, hidden = self.lstm(
            embds.view(len(sentence), 1, -1),
            self.hidden
        )
        tag_space = self.hidden2tag(out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def prepare_sentence(sentence, to_idx):
    idxs = [to_idx[w] for w in sentence]
    return Variable(torch.LongTensor(idxs))

model = LSTMTagger(len(word_to_idx), 100, 128, len(tag_to_idx))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

for epoch in range(300):
    for sentence, tag in training_data:
        model.hidden = model.init_hidden()
        input = prepare_sentence(sentence, word_to_idx)
        targets = prepare_sentence(tag, tag_to_idx)
        tag_scores = model(input)

        loss = criterion(tag_scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

input = prepare_sentence(training_data[0][0], word_to_idx)
tag_scores = model(input)

print(training_data[0][0])
print(input)
print(tag_scores)
print(tag_to_idx)


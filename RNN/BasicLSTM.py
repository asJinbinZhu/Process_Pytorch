import torch
from torch.autograd import Variable
import torch.nn.functional as F

# prepare data
def prepare_sequence(seq,to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

training_data = [
    ('The dog ate the apple'.split(),['DET','NN','V','DET','NN']),
    ('Everybody read that book'.split(),['NN','V','DET','NN'])
]

word_to_ix = {}

for sent,tags in training_data:
    for word in sent:
        if word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

print(word_to_ix)
tag_to_ix = {'DET':0,'NN':1,'V':2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(torch.nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size):
        super(LSTMTagger,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = torch.nn.Embedding(vocab_size,embedding_dim)

        self.lstm = torch.nn.LSTM(embedding_dim,hidden_dim)

        self.hidden2tag = torch.nn.Linear(hidden_dim,tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1,1,self.hidden_dim)),
                Variable(torch.zeros(1,1,self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out,self.hidden = self.lstm(
            embeds.view(len(sentence),1,-1),
            self.hidden
        )

        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        tag_scores = F.log_softmax(tag_space,dim=1)

        return tag_scores

model = LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_ix),len(tag_to_ix))
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

inputs = prepare_sequence(training_data[0][0],word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(1000):
    for sentence,tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()

        sentence_in = prepare_sequence(sentence,word_to_ix)
        targets = prepare_sequence(tags,tag_to_ix)

        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores,targets)
        loss.backward()
        optimizer.step()

inputs = prepare_sequence(training_data[0][0],word_to_ix)
tag_scores = model(inputs)

print(tag_scores)
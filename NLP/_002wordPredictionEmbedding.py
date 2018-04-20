import torch
import torch.nn.functional as F
from torch.autograd import Variable

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """
                When forty winters shall besiege thy brow, 
                And dig deep trenches in thy beauty's field, 
                Thy youth's proud livery, so gazed on now, 
                Will be a tatter'd weed, of small worth held: 
                Then being ask'd where all thy beauty lies, 
                Where all the treasure of thy lusty days, 
                To say, within thine own deep-sunken eyes, 
                Were an all-eating shame and thriftless praise. 
                How much more praise deserved thy beauty's use, 
                If thou couldst answer 'This fair child of mine 
                Shall sum my count and make my old excuse,' 
                Proving his beauty by succession thine! 
                This were to be new made when thou art old, 
                And see thy blood warm when thou feel'st it cold.
                """.split()

trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2]) for i in range(len(test_sentence)-2)]

vocb = set(test_sentence)
word_to_idx = {word:i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]:word for word in word_to_idx}

class nGramModel(torch.nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(nGramModel,self).__init__()
        self.n_word = vocb_size
        self.embedding = torch.nn.Embedding(self.n_word, n_dim)
        self.linear1 = torch.nn.Linear(context_size*n_dim, 128)
        self.linear2 = torch.nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob

model = nGramModel(len(word_to_idx), CONTEXT_SIZE, EMBEDDING_DIM)

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# training
for epoch in range(100):
    print('epoch{}'.format(epoch+1))
    print('*'*10)
    running_loss = 0
    for data in trigram:
        word, label = data
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))

        # forward
        out = model(word)
        loss = criterion(out, label)
        running_loss += loss.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Loss:{:.6f}'.format(running_loss/len(word_to_idx)))

word, label = trigram[5]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = model(word)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.data[0]]
print('real word is {}, predict word is {}'.format(label, predict_word))


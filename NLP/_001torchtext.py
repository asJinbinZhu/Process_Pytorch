'''
Q: You are using pip version 9.0.1, however version 10.0.1 is available.
A: go to https://pypi.org/project/pip/ download latest version
    sudo tar -zxvf pip-10.0.1.tar.gz -C /home/admin/
    sudo python setup.py install

Q: how to install torchtext
A: go to https://github.com/pytorch/text
    pip install torchtext
'''
from torchtext import data
import spacy

spacy.en = spacy.load('en')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

train, val, test = data.TabularDataset.split(
    path = './data',
    train = 'train.tsv',
    validation = 'val.tsv',
    test = 'test.tsv',
    format = 'tsv',
    fields = [('Test', TEXT), ('Label', LABEL)]
)

TEXT.build_vocab(train, vectors = 'glove.6B.100d')
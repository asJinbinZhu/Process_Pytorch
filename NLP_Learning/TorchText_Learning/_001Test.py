import torchtext.data as data
import spacy
spacy_en = spacy.load('en')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

train, val, test = data.TabularDataset.split(
    path = './data',
    train = 'train.tsv',
    val = 'val.tst',
    test = 'test.tsv',
    format = 'tsv',
    fields = [('Text', TEXT), ('Label', LABEL)]
)

TEXT.build_vocab(train, vectors = 'glove.6B.100d')

train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), sort_key = lambda x: len(x.Text),
    batch_sizes = (32, 256, 256),
    device = -1
)

vocab = TEXT.vocab
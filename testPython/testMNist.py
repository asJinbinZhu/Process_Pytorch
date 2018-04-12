import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

DOWNLOAD = True
BATCH_SIZE=64

train_data = dsets.MNIST(
    root='../mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD,
)
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i'%train_data.train_labels[0])
plt.show()

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data = dsets.MNIST(
    root='../mnist',
    train=False
)

for step,(x,y) in enumerate(train_loader):
    print(step)
import torch
import torch.utils.data as Data

torch.manual_seed(1)
BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

# torch ---> dataset
torch_dataset = Data.TensorDataset(data_tensor=x,target_tensor=y)
# dataset --->DataLoader
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2,
)

# train all data for 3 times
for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        # train your data
        print('Epoch:',epoch,'|Step:',step,'|batch x:',batch_x.numpy(),'|batch y:',batch_y.numpy())
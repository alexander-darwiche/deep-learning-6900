####################################################
# First of the first, please start writing it early!
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torch.optim as optim

# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

batch_size = 10

train_loader = Data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=torchvision.transforms.ToTensor())

test_loader = Data.DataLoader(dataset=test_data,  batch_size = batch_size, shuffle=True )

# ----------------- build the model ------------------------
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.linear1 = nn.Linear(100,100)
        self.linear2 = nn.Linear(100,100)
        self.linear3 = nn.Linear(100,100)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x

model = net()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = .001, momentum = .9)

for epoch in range(...):
    for step, (input, target) in enumerate(train_loader):
        model.train()   # set the model in training mode
            ...
            ...

        if step % 50 == 0:
            test()
            ...
            ...
            save_model()
            ...
            ...


# ------ maybe some helper functions -----------
def test(......):
    model.eval()  # switch the model to evaluation mode
    ...
    ...
    print('Test set: Epoch[{}]:Step[{}] Accuracy: {}% ......'.format(...))


def save_model(......):
    ...
    torch.save(model.state_dict(), "./model/xxxx.pt".format(...))
    ...
    # when get a better model, you can delete the previous one
    os.remove(......)   # you need to 'import os' first
    ...


def load_model(model, ......):
    ...
    model.load_state_dict(torch.load("./model/xxxx.pt"))
    ...


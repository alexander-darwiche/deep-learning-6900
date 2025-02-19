####################################################
# First of the first, please start writing it early!
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torch.optim as optim
import os

# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

batch_size = 128

train_loader = Data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=torchvision.transforms.ToTensor())

test_loader = Data.DataLoader(dataset=test_data,  batch_size = batch_size, shuffle=True )

# ----------------- build the model ------------------------
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(32*32*3,2000)
        self.fc2 = nn.Linear(2000,1000)
        self.fc3 = nn.Linear(1000,500)
        self.fc4 = nn.Linear(500,100)
        self.fc5 = nn.Linear(100,10)
        self.bn1 = nn.BatchNorm1d(2000)
        self.bn2 = nn.BatchNorm1d(100)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.bn2(x)
        x = self.fc5(x)
        return x

# ------ maybe some helper functions -----------
def test(epoch, step):
    model.eval()  # switch the model to evaluation mode
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test set: Epoch[{}]:Step[{}] Accuracy: {}% ......'.format(epoch, step, 100 * correct // total))



# def save_model(......):
#     ...
#     torch.save(model.state_dict(), "./model/xxxx.pt".format(...))
#     ...
#     # when get a better model, you can delete the previous one
#     os.remove(......)   # you need to 'import os' first
#     ...


# def load_model(model, ......):
#     ...
#     model.load_state_dict(torch.load("./model/xxxx.pt"))
#     ...



model = net()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = .001, momentum = .9)
epochs = 10


for epoch in range(epochs):
    total_loss = 0
    for step, (input, target) in enumerate(train_loader):
        model.train()   # set the model in training mode
        optimizer.zero_grad() # Set all gradients back to 0
        output = model(input) # Forward pass through the model
        loss = loss_func(output,target) # Evaluate the loss/difference from ground truth
        loss.backward() # Calculate gradients to back propogate
        optimizer.step() # Back-propogate the losses
        total_loss += loss.item() # Accumulate losses

        if step % 50 == 0:
            print('Step: ',step)
            test(epoch, step)
            # ...
            # ...
            # save_model()
            # ...
            # ...
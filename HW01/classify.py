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
def test_accuracy():
    model.eval()  # switch the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (100 * correct // total)



def save_model(test_accuracy):

    model_path = "./model/model.pt"

    if os.path.exists(model_path):
        model2 = torch.load(model_path)
        if model2['test_accuracy'] < test_accuracy:
            print('Model Updated')
            os.remove(model_path)
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_accuracy
            }, model_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_accuracy': test_accuracy
        }, model_path)

def load_model():
    model = net()
    model.load_state_dict(torch.load("./model/model.pt")['model_state_dict'])
    return model

def test():
    print('hi')



import sys

if 'train' in sys.argv[1:]:
    model = net()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = .001, momentum = .9)
    epochs = 10

    results = {"Epoch", "Step", "Train_Loss","Train_Acc","Test_Loss","Test_Acc"}
        
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
            _, predicted = output.max(1)

            if step % 50 == 0:
                test_acc = test_accuracy(epoch, step)
                correct = 0
                total = 0
                total += target.size(0)
                correct += (predicted == target).sum().item()
                train_acc =  100 * correct // total
                
            
                save_model(test_acc)

elif 'test' in sys.argv[1:]:
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize image to 32x32
        transforms.ToTensor(),        # Convert image to tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR-10 normalization
    ])

    # Load image from file
    image_path = sys.argv[2]  # Replace with your file path
    image = Image.open(image_path).convert("RGB")

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, 3, 32, 32)

    model = load_model()

    # Run inference
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        predicted_class = output.argmax(1).item()

    # CIFAR-10 Class Labels
    labels_map = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }

    print(f"prediction result: {labels_map[predicted_class]}")

elif 'load_pic' in sys.argv[1:]:
    
    import matplotlib.pyplot as plt
    image_tensor, label = test_data[1] 

    # Define normalization values used for CIFAR-10
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    # Denormalize if necessary
    def denormalize(tensor, mean, std):
        return tensor * std + mean  # Reverse normalization

    # Plot image
    # plt.imshow(denormalize(image_tensor, mean, std).permute(1, 2, 0))
    # plt.title("Input Image")
    # plt.show()

    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])

    # Save first 5 images as PNG
    for i in range(10):  
        image_tensor, label = test_data[i]  # Get image tensor
        image_tensor = denormalize(image_tensor, mean, std)  # Denormalize
        
        # Convert tensor to PIL image
        image_pil = transforms.ToPILImage()(image_tensor)
        
        # CIFAR-10 Class Labels
        labels_map = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
        # Save as PNG
        image_pil.save(f"cifar10_{labels_map[label]}_{i}.png")
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
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import pandas as pd


# Normalize the CIFAR10 Dataset (from Pytorch Website)
transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize image to 32x32
        transforms.ToTensor(),        # Convert image to tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR-10 normalization
    ])
# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=transform,    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

# Define a batch size to set with, using 128
batch_size = 128

# Load the training data from the dataset, breaking it into batches
train_loader = Data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=transform)

# Load the training data from the dataset, breaking it into batches
test_loader = Data.DataLoader(dataset=test_data,  batch_size = batch_size, shuffle=True )

# ----------------- build the model ------------------------

# Define a Deep Nueral Network with only Fully Connected Layers or Batch Normalization
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(15680,2000)
        self.fc2 = nn.Linear(2000,10)
        self.bn1 = nn.LayerNorm(2000)
        self.bn2 = nn.LayerNorm(10)
        self.bn3 = nn.BatchNorm2d(10)
        self.bn4 = nn.BatchNorm2d(20)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 3)

    # Define a forward pass through the model
    def forward(self, x):
        x = F.relu(self.bn3(self.conv1(x)))
        x = F.relu(self.bn4(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten before entering FC layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x

# ------ maybe some helper functions -----------
def test_accuracy():
    '''
    This function will return the accuracy of the model
    on the testing data.
    
    '''
    model.eval()  # switch the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, target = data # split data into images/target
            outputs = model(images) # Determine the "prediction" or "output" from the model based on the images
            loss = loss_func(outputs,target) # Find the loss (difference between ground truth and prediction)
            _, predicted = torch.max(outputs, 1) # Determine the most likely label for the image
            total += target.size(0) # Find the amount of total images in batch
            correct += (predicted == target).sum().item() # Find the total images that are correctly labeled
    # Return Accuracy and Loss
    return [(100 * correct / total),loss.item()]



def save_model(test_accuracy):
    '''
    This function will save the model in the event that it exceeds
    a previous model's test accuracy, or if a model does not exist.

    '''

    model_path = "./model/model.pt" # Where to save model and what to name it.

    # Determine if a model already exists, save model if not
    if os.path.exists(model_path):
        model2 = torch.load(model_path) # Load the model from the save. This isn't exactly the "load_state_dict" as I added test_accuracy to the model save
        if model2['test_accuracy'] < test_accuracy: # If the "new model" has better test accuracy, then continue
            os.remove(model_path) # Remove the old model
            torch.save({ # save the new model parameters AND its test accuracy
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_accuracy
            }, model_path)
    else:
        torch.save({ # save the new model parameters AND its test accuracy
            'model_state_dict': model.state_dict(),
            'test_accuracy': test_accuracy
        }, model_path)

def load_model():
    '''
    This function loads the model_state_dict or model state dictionary from the saved
    model.

    '''
    model = net()
    model.load_state_dict(torch.load("./model/model.pt")['model_state_dict'])
    return model




# This if statement dictates the interactions depending on the arugments passed to command line.
# IF 'Train', train the model.
# IF 'Test' or 'Predict', try to predict the command line image's class
if 'train' in sys.argv[1:]:
    model = net()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = .001, momentum = .9, weight_decay=0.05)
    epochs = 10
    
    # Print the Columns for the results table
    results = pd.DataFrame({"Epoch": [], "Step": [], "Train_Loss": [], "Train_Acc": [], "Test_Loss": [], "Test_Acc": []})
    print(" | ".join("{:<10}".format(col) for col in results.columns.to_list()))

    for epoch in range(epochs):

        for step, (input, target) in enumerate(train_loader):
            train_loss = 0
            model.train()   # set the model in training mode
            optimizer.zero_grad() # Set all gradients back to 0
            output = model(input) # Forward pass through the model
            loss = loss_func(output,target) # Evaluate the loss/difference from ground truth
            loss.backward() # Calculate gradients to back propogate
            optimizer.step() # Back-propogate the losses
            train_loss += loss.item() # Accumulate losses
            _, predicted = output.max(1)

            if step == 0:
                test_acc, test_loss = test_accuracy() # Get test accuracy and loss
                correct = 0 
                total = 0
                total += target.size(0)
                correct += (predicted == target).sum().item() # Calculate train accuracy and loss, exactly as done for test, just on the training data.
                train_acc =  100 * correct / total
                
                # Create the new row of Results Dataframe
                new_row = pd.DataFrame([{"Epoch": epoch, "Step": step, "Train_Loss": round(train_loss, 2), "Train_Acc": round(train_acc, 2), "Test_Loss": round(test_loss, 2), "Test_Acc": round(test_acc, 2)}])
                results = pd.concat([results,new_row], ignore_index=True)

                # Print the new row of data from the first step of the new epoch
                print(" | ".join("{:<10}".format(str(value)) for value in results.iloc[-1].values))

                # Attempt to save model if needed
                save_model(test_acc)

elif 'predict' in sys.argv[1:] or 'test' in sys.argv[1:]:
    
    # Need to "normalize" the picture that is being input, ensuring its the correct size
    # and the channels are normalized based on mean/std.
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
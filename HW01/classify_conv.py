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

# Define the data augmentation transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Random crop and resize
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation by 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Normalize with standard ImageNet values
])

# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=train_transform,    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
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
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)  # Reduced number of filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Reduced number of filters
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(14400, 1000)  # Adjust based on input image size after convolutional layers
        self.fc2 = nn.Linear(1000, 10)

        # Dropout Layers
        self.dropout = nn.Dropout(p=0.1)  # Randomly drop 50% of the neurons

    def forward(self, x):
        # Convolutional Layers with Batch Normalization and ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # Add max-pooling after convolutions to reduce spatial size
        
        # Flatten before passing to fully connected layers
        x = torch.flatten(x, 1)  # Flatten the output from conv layers into a vector

        # Fully Connected Layers with Batch Normalization, ReLU, and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Final output layer (no activation here, typically for classification)
        
        return x

# ------ maybe some helper functions -----------
def test_accuracy(model, test_loader, loss_func, device='cpu'):
    '''
    This function will return the accuracy of the model
    on the testing data.
    
    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the testing dataset.
        loss_func: Loss function used for evaluation.
        device: The device to run the model on ('cpu' or 'cuda').

    Returns:
        Tuple: (test accuracy %, test loss)
    '''
    model.eval()  # Switch the model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0  # Initialize total loss
    
    with torch.no_grad():  # Disable gradient computation for inference
        for images, target in test_loader:
            # Move data to the correct device
            images, target = images.to(device), target.to(device)
            
            outputs = model(images)  # Forward pass through the model
            loss = loss_func(outputs, target)  # Compute the loss
            
            total_loss += loss.item() * target.size(0)  # Accumulate the weighted loss
            _, predicted = outputs.max(1)  # Get predicted class (the one with highest probability)
            total += target.size(0)  # Accumulate the total number of samples
            correct += (predicted == target).sum().item()  # Count correct predictions
    
    # Compute final accuracy and average loss
    accuracy = 100 * correct / total
    avg_loss = total_loss / total  # Average loss over all samples

    return accuracy, avg_loss



def save_model(test_accuracy):
    '''
    This function will save the model in the event that it exceeds
    a previous model's test accuracy, or if a model does not exist.

    '''

    model_path = "./model/model_conv.pt" # Where to save model and what to name it.

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
    import pdb;pdb.set_trace()
    model.load_state_dict(torch.load("./model/model_conv.pt")['model_state_dict'])
    return model


def compute_train_accuracy(model, train_loader, loss_func, device='cpu'):
    """
    Computes the accuracy and average loss of the model on the training dataset.
    
    Args:
        model: The trained PyTorch model.
        train_loader: DataLoader for the training dataset.
        loss_func: Loss function used for evaluation.
        device: 'cpu' or 'cuda' depending on available hardware.
    
    Returns:
        Tuple: (accuracy %, average loss)
    """
    model.train()  # Set model to training mode
    correct = 0
    total = 0
    total_loss = 0  # Track cumulative loss
    
    with torch.no_grad():  # Disable gradient computation
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)  # Move to device
            
            outputs = model(images)  # Forward pass
            loss = loss_func(outputs, target)  # Compute batch loss
            total_loss += loss.item() * target.size(0)  # Accumulate weighted loss
            
            _, predicted = outputs.max(1)  # Get predicted class
            total += target.size(0)  # Update total count
            correct += (predicted == target).sum().item()  # Count correct predictions
    
    # Compute final accuracy and average loss
    accuracy = 100 * correct / total
    avg_loss = total_loss / total  # Average loss over all samples
    
    return accuracy, avg_loss


# Intermediate Results outputs
activations = []
    
def hook_fn(module, input, output):
    activations.append(output)

# This if statement dictates the interactions depending on the arugments passed to command line.
# IF 'Train', train the model.
# IF 'Test' or 'Predict', try to predict the command line image's class
if 'train' in sys.argv[1:]:
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import torch_directml

    # # Set the device to DirectML
    # device = torch_directml.device()
    
    # Initialize model, loss function, and optimizer, and move model to the device
    model = net().to(device)  
    loss_func = nn.CrossEntropyLoss()
    
    # model.conv1.register_forward_hook(hook_fn)
    # model.conv2.register_forward_hook(hook_fn)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    epochs = 50
    
    # Create an empty dataframe for results
    results = pd.DataFrame({"Epoch": [], "Step": [], "Train_Loss": [], "Train_Acc": [], "Test_Loss": [], "Test_Acc": []})
    print(" | ".join("{:<10}".format(col) for col in results.columns.to_list()))

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0
        
        
        for step, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)  # Move data to the device
            model.train()  # Set model in training mode
            optimizer.zero_grad()  # Reset gradients
            output = model(input)  # Forward pass
            loss = loss_func(output, target)  # Compute loss
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update weights
            
            train_loss += loss.item() * target.size(0)  # Accumulate weighted loss
            _, predicted = output.max(1)  # Get predicted class
            total += target.size(0)
            correct += (predicted == target).sum().item()  # Count correct predictions
            
        # Get test accuracy and loss (ensure model is in eval mode for testing)
        test_acc, test_loss = test_accuracy(model, test_loader, loss_func, device)
        
        # Compute training accuracy (after the epoch) and loss
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / total

        # Create a new row in the results dataframe
        new_row = pd.DataFrame([{
            "Epoch": epoch, 
            "Step": step, 
            "Train_Loss": round(avg_train_loss, 2),
            "Train_Acc": round(train_acc, 2), 
            "Test_Loss": round(test_loss, 2), 
            "Test_Acc": round(test_acc, 2)
        }])
        results = pd.concat([results, new_row], ignore_index=True)

        # Print the results from the first step of the new epoch
        print(" | ".join("{:<10}".format(str(value)) for value in results.iloc[-1]))


        # Optionally save the model
        save_model(test_acc)
    #     fig, axes = plt.subplots(1, len(activations), figsize=(10, 5))
    # for i, activation in enumerate(activations):
    #     axes[i].imshow(activation[0, 0].detach().numpy(), cmap='viridis') # Display first channel of the first image in the batch
    #     axes[i].set_title(f"Layer {i+1}")
    #     axes[i].axis('off')
    # plt.show()

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
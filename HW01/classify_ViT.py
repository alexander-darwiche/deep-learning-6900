import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

# Define a basic Vision Transformer (ViT) block

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=emb_dim,
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # Apply convolution to extract patches and embed them into a higher-dimensional space
        x = self.conv(x)
        x = x.flatten(2)  # Flatten the patches (batch_size, emb_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, emb_dim)
        return x


class ViT(nn.Module):
    def __init__(self, num_classes=10, patch_size=4, emb_dim=256, num_heads=8, num_layers=6):
        super(ViT, self).__init__()
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_dim=emb_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, (32 // patch_size) * (32 // patch_size), emb_dim))  # CIFAR10 images are 32x32
        
        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # Extract patches and embed them
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Pass through the transformer encoder
        x = self.transformer(x)
        
        # Class token is typically used for classification, but here we use the average of the transformer outputs
        x = x.mean(dim=1)  # (batch_size, emb_dim)
        
        # Final classification layer
        x = self.fc(x)
        return x

# Define data transforms for training and testing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
model = ViT(num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch_directml

# Set the device to DirectML
device = torch_directml.device()
model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

# Evaluate on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

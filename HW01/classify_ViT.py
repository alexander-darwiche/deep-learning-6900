import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# Check if DirectML is available
import torch_directml
device = torch_directml.device()
print(f"Using device: {device}")

# Patch Embedding Layer
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size=32):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Convolution to extract patches
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, emb_dim)
        return x

# Vision Transformer (ViT-H) Model
class ViT(nn.Module):
    def __init__(self, num_classes=10, patch_size=4, emb_dim=512, num_heads=12, num_layers=12, dropout=0.1):
        super(ViT, self).__init__()
        
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_dim=emb_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, (32 // patch_size) ** 2, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=4 * emb_dim, dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(emb_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.patch_embedding(x)  # Patch embedding
        x = x + self.pos_embedding  # Add positional encoding
        x = self.dropout(x)

        x = self.transformer(x)  # Transformer encoder
        x = self.norm(x.mean(dim=1))  # Global average pooling over patches
        x = self.fc(x)  # Classification layer
        return x

# Data Augmentation (Essential for ViTs)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Initialize Model
model = ViT(num_classes=10).to(device)

# Define Loss & Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# Training Loop
epochs = 20
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
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

# Evaluate Model on Test Set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

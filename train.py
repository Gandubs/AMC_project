import torch
from torch.utils.data import Dataset
import numpy as np

# 0: Analog, 1: FSK, 2: PAM, 3: PSK, 4: QAM
family_map = {
    # Analog
    4:0, 14:0, 24:0, 33:0, 44:0, 54:0,
    # FSK
    2:1, 12:1, 22:1, 32:1,
    # PAM
    3:2, 13:2, 23:2,
    # PSK
    0:3, 10:3, 20:3, 30:3, 40:3, 50:3,
    # QAM
    1:4, 11:4, 21:4, 31:4, 41:4, 51:4, 61:4
}

class HisarModDataset(Dataset):
    def __init__(self, data_path, label_path, family_map, transform=None):
        self.data = np.load(data_path)          # Shape: [N, 2, 1024]
        self.labels = np.load(label_path)       # Shape: [N,] → modulation type (0–25)
        self.transform = transform
        self.family_map = family_map            # Dict: mod_type → family

        # Convert modulation type labels to family labels (0–4)
        self.family_labels = np.array([self.family_map[label] for label in self.labels])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # [2, 1024]
        y = self.family_labels[idx]  # Family label: 0–4

        if self.transform:
            x = self.transform(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

from torch.utils.data import random_split, DataLoader

# Load the full dataset
dataset = HisarModDataset('train_data_reshaped_1024.npy', 'train_labels_continuous.npy', family_map)

# Split sizes as per the paper: 8/15 train, 2/15 val, 5/15 test
total = len(dataset)
train_size = int(8/15 * total)
val_size = int(2/15 * total)
test_size = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
import torch

# Create dictionary to store data & labels
test_data = []
test_labels = []

for x, y in test_dataset:
    test_data.append(x)
    test_labels.append(y)

test_data = torch.stack(test_data)       # Shape: [N, 2, 1024]
test_labels = torch.stack(test_labels)   # Shape: [N]

# Save both as a tuple or dict
torch.save({'data': test_data, 'labels': test_labels}, "test_dataset.pt")

print("✅ Validation dataset saved to test_dataset.pt")


import torch
import torch.nn as nn
import torch.nn.functional as F

class HisarModCNN(nn.Module):
    def __init__(self):
        super(HisarModCNN, self).__init__()

        self.noise = nn.Identity()  # Placeholder for SNR-based noise injection

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1,3), padding=(0,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=(1,3), padding=(0,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=(1,3), padding=(0,1))
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout4 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 classes: Analog, FSK, PAM, PSK, QAM

    def forward(self, x):
        # Input x: [batch, 2, 1024]
        x = x.unsqueeze(1)  # Convert to [batch, 1, 2, 1024]

        x = self.noise(x)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Instantiate model
model = HisarModCNN()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, criterion, optimizer, epochs=20, save_path="hisarmod_cnn.pth"):

    model = model.to(device)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation after each epoch
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved model checkpoint at epoch {epoch+1} with val_acc {val_acc:.4f}")

    print("Training complete. Best validation accuracy:", best_val_acc)

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc


model = HisarModCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
if __name__ == "__main__":
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=10, save_path="hisarmod_best_model.pth")

    # Optionally, you can also test the model after training
    from test import test_model
    test_model(model, test_loader, model_path="hisarmod_best_model.pth")






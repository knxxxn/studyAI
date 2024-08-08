#컨볼루션 네트워크를 구현하여 MNIST를 분류
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set hyperparameters
num_epochs = 3
batch_size = 100
learning_rate = 0.001
# Load MNIST dataset and preprocessing
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(
root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(
root='./data', train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.view(-1, 64 * 7 * 7)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x

model = SimpleCNN()

# Model training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    outputs = model(images) # forward-propagation
    loss = criterion(outputs, labels)
    optimizer.zero_grad() # back-propagation
    loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], 'f'Step [{i+1}/{total_step}], 'f'Loss: {loss.item():.4f}')

# Model testing
model.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print(f'Accuracy: {100 * correct / total}%')
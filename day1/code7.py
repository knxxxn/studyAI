#MLP와 ReLU 함수, Adam을 이용하여 MNIST 분류기 학습
'''
시그모이드(sigmoid) 함수
시그모이드 함수는 미분값이 대체적으로 작음
입력값이 큰 양수/음수일 때, 기울기가 0에 매우 가까워짐
뉴럴 네트워크에서 그래디언트 소실(gradient vanishing) 문제 발생 가능

ReLU(Rectified Linear Unit) 함수
ReLU 함수나, 그의 변형인 Leaky ReLU, GELU 함수를 이용하여 학습을 수행할 수 있음

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set hyperparameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Load MNIST dataset and preprocessing
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # instead of Softmax
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MLP(input_size, hidden_size, num_classes)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # instead of SGD
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Vectorize the MNIST image
        images = images.reshape(-1, input_size)

        # Forward-propagation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Back-propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], 'f'Step [{i+1}/{total_step}], 'f'Loss: {loss.item():.4f}')

# Testing the trained model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f'Accuracy: {100 * correct / total}%')
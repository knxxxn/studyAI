'''
VGG
VGG는 Oxford VGG에서 만든 딥 컨볼루션 네트워크임
(3, 224, 224) 입력을 기준으로 제작되어 있음
PyTorch의 torchvision에서는 VGG-11, 13, 16, 19를 제공함
제공된 VGG는 구조만 가져와서 사용할 수도 있고, 미리 학습된 네트워크도 사용 가능
'''
import torch
import torchvision.models as models
model = models.vgg16(pretrained=True)

'''
ResNet
컨볼루션 네트워크가 점점 더 깊어지면서, 학습 등에서 어려움이 있었음
ResNet에서는 잔차 연결(residual connection)을 제안하여 이러한 문제를 해결함
(3, 224, 224) 입력을 기준으로 제작되어 있음
PyTorch의 torchvision에서는 ResNet-18, 34, 50, 101, 152를 제공함
제공된 ResNet은 구조만 가져와서 사용할 수도 있고, 미리 학습된 네트워크도 사용 가능
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation
transform = transforms.Compose([
transforms.Resize(224), # ResNet18 is built for 224 x 224 size
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Load ResNet-18 and modify it
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # CIFAR-10 has 10 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define training function
def train(epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

# Define testing function
def evaluate():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on 10000 test images: {100 * correct / total:.2f}%')

# Model training with 5 epochs
train(5)
# Model testing
evaluate()
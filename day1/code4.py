#MNIST 데이터셋(손글씨 숫자 데이터셋)
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Download MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

image, label = train_dataset[0] # sample
print(image.shape)
print(label)

plt.imshow(image.squeeze())
plt.show()

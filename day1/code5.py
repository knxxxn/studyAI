#퍼셉트론
import torch
import torch.nn as nn
import torch.optim as optim

#AND 연산
# Data preparation
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_and = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# Define Perceptron model
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Training AND operation
model_and = Perceptron() #실제화
criterion = nn.BCELoss() #목적함수, 손실함수(binary cross-entropy)
optimizer_and = optim.SGD(model_and.parameters(), lr=0.1) #lr = learning rate, 숫자가 클수록 반영을 많이 한다

for epoch in range(1000):
    optimizer_and.zero_grad() #초기화
    output = model_and(X)
    loss = criterion(output, y_and)
    loss.backward() #back propagation
    optimizer_and.step()

# Testing the results
print(model_and(X).round())

#OR 연산
# Data preparation
y_or = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)

# Training OR operation
model_or = Perceptron()
optimizer_or = optim.SGD(model_or.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer_or.zero_grad()
    output = model_or(X)
    loss = criterion(output, y_or)
    loss.backward()
    optimizer_or.step()
# Testing the results
print(model_or(X).round())

#XOR 연산
#선형 분류가 불가능함, 학습 실패
#단일(1층짜리) 퍼셉트론 = 선형 분류기
# Data preparation
y_xor = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Training XOR operation
model_xor = Perceptron()
optimizer_xor = optim.SGD(model_xor.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer_xor.zero_grad()
    output = model_xor(X)
    loss = criterion(output, y_xor)
    loss.backward()
    optimizer_xor.step()

# Testing the results
print(model_xor(X).round())
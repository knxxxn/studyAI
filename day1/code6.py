#다층 퍼셉트론
#비선형 분류 성공, 학습 성공
'''
다층 퍼셉트론(multi-layer perceptron; MLP)
퍼셉트론이 여러 층 반복되어 있는 구조
층이 여러 번 반복되며, 데이터를 좀 더 다양한 방식으로 표현 가능
선형이 아닌, 비선형 방식의 분류기를 학습할 수 있게 됨
'''
import torch
import torch.nn as nn
import torch.optim as optim

from day1.code5 import criterion, X, y_xor

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 2) # input -> hidden
        self.output = nn.Linear(2, 1) # hidden -> output
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Training XOR again
model_xor = MLP()
optimizer_xor = optim.SGD(model_xor.parameters(), lr=0.1)
for epoch in range(15000):
    optimizer_xor.zero_grad()
    output = model_xor(X)
    loss = criterion(output, y_xor)
    loss.backward()
    optimizer_xor.step()

# Testing the results
print(model_xor(X).round())
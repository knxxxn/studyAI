#RNN과 LSTM으로 신호 예측
import torch
import torch.nn as nn
# Define RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out

# Set hyperparameters
input_size = 1
hidden_size = 10
output_size = 1
seq_length = 5

# Model initialization
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Generate dataset
x = torch.randn(10, seq_length, input_size)
y = torch.sum(x, dim=1)

# Model training
for epoch in range(100):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Model testing
test_input = torch.randn(1, seq_length, input_size)
predicted = model(test_input)
print(f"Test input sum: {test_input.sum().item():.4f}")
print(f"Predicted sum: {predicted.item():.4f}")

# Define LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, (hidden, _) = self.lstm(x) # LSTM returns hidden state and cell state
        out = self.fc(hidden.squeeze(0))
        return out

# Set hyperparameters
input_size = 1
hidden_size = 10
output_size = 1
seq_length = 5

# Model initialization
model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Generate dataset
x = torch.randn(10, seq_length, input_size)
y = torch.sum(x, dim=1)

# Model training
for epoch in range(100):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Model testing
test_input = torch.randn(1, seq_length, input_size)
predicted = model(test_input)
print(f"Test input sum: {test_input.sum().item():.4f}")
print(f"Predicted sum: {predicted.item():.4f}")
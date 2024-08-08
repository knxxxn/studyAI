'''
리커런트 뉴럴 네트워크(recurrent neural network; RNN)
순차적(sequential)으로 생성되는 데이터(문장, 매일의 날씨 등)를 표현하기 위해
제안된 뉴럴 네트워크
pytorch에서는 RNN을 제공함
'''
import torch
import torch.nn as nn
import numpy as np
input_size = 4
hidden_size = 2

# 1-hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

input_data = np.array([[h, e, l, l, o],
[e, o, l, l, l],
[l, l, e, e, l]], dtype=np.float32)

# Transform the numpy array into torch tensor
input_data = torch.tensor(input_data)
print(f'Input size: {input_data.shape}')
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
outputs, _status = rnn(input_data)
print(f'Output size: {outputs.shape}')

#그래디언트 소실/폭발 문제를 해소하기 위해, LSTM, GRU를 사용함

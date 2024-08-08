'''
컨볼루션(convolution)
입력 데이터 위에서 stride만큼 filter(kernel)을 이동시키면서, 겹쳐지는 부분의 각
원소의 값을 곱해서 모두 더한 값을 출력으로 하는 연산

Stride
컨볼루션 연산에서 filter를 한 번에 얼마나 이동할 것인지를 지정함

패딩(padding)
패딩을 이용하여 입력 데이터의 크기를 보전하기도 함

PyTorch에서는 다양한 차원의 컨볼루션 연산을 지원함
입력의 (N, C, H, W)는 각각 배치 사이즈, 입력 채널 수, 높이, 너비를 의미함

'''
import torch
import torch.nn as nn

t = torch.rand(size=(5, 1, 227, 227))
conv = nn.Conv2d(in_channels=1,
out_channels=1,
kernel_size=(11, 11),
stride=(4, 4),
padding=(0, 0))

print(conv(t).shape)

'''
풀링(pooling)
컨볼루션 네트워크는 때때로 풀링 연산을 추가하여 입력 데이터의 크기를 줄임
주로 최대 풀링(max pooling)과 평균 풀링(average pooling)이 사용됨
'''
t = torch.rand(size=(5, 1, 227, 227))
conv = nn.Conv2d(in_channels=1,
out_channels=1,
kernel_size=(11, 11),
stride=(4, 4),
padding=(0, 0))

pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
print(pool(conv(t)).shape)
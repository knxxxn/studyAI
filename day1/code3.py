import numpy as np
import torch

# View (reshape)
t = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
print(t.shape)
print(t.view([-1, 3])) # '-1' for automatic calculation
print(t.view([-1, 3]).shape)
print(t.view([-1, 1, 3]))
print(t.view([-1, 1, 3]).shape)
print(t.view([-1, 2, 3]))

# Squeeze
t = torch.tensor([[0], [1], [2], [3]])
print(t)
print(t.shape)
print(t.squeeze()) #행렬에서 불필요한 괄호 없앨 때 사용
print(t.squeeze().shape)

# Concatenation -> 문자열 더하기
t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])
print(torch.cat([t1, t2], dim=0)) # 4 x 2
print(torch.cat([t1, t2], dim=1)) # 2 x 4

# Stacking
t1 = torch.tensor([1, 4])
t2 = torch.tensor([2, 5])
t3 = torch.tensor([3, 6])
print(torch.stack([t1, t2, t3])) # 3 x 2
print(torch.stack([t1, t2, t3], dim=1)) # 2 x 3

#cat을 이용하여 stack과 같은 결과 만들기
t1 = torch.tensor([1, 4])
t2 = torch.tensor([2, 5])
t3 = torch.tensor([3, 6])

# Stack along the first dimension (dim=0)
t1_unsq = t1.unsqueeze(0)  # Shape: (1, 2)
t2_unsq = t2.unsqueeze(0)  # Shape: (1, 2)
t3_unsq = t3.unsqueeze(0)  # Shape: (1, 2)
stacked_dim0 = torch.cat([t1_unsq, t2_unsq, t3_unsq], dim=0)
print(stacked_dim0)  # 3 x 2

# Stack along the second dimension (dim=1)
t1_unsq = t1.unsqueeze(1)  # Shape: (2, 1)
t2_unsq = t2.unsqueeze(1)  # Shape: (2, 1)
t3_unsq = t3.unsqueeze(1)  # Shape: (2, 1)
stacked_dim1 = torch.cat([t1_unsq, t2_unsq, t3_unsq], dim=1)
print(stacked_dim1)  # 2 x 3

# Ones and Zeros
t = torch.tensor([[3, 1, -2], [5, 0, 4]])
print(t.shape)
print(torch.ones_like(t))
print(torch.zeros_like(t))

# In-place Operation
t = torch.tensor([[1., 2.], [3., 4.]])
print(t.mul(2.)) #t에 2곱하기
print(t)
print(t.mul_(2.)) #t에 2곱하고 저장하기
print(t)


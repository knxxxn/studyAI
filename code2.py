# torch-based Tensor
import torch

t = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(f'Tensor values: \n{t}')
print(f't[0]: {t[0]}, t[-1]: {t[-1]}')
print(f't[:2]: {t[:2]}')
print(f't[1:]: {t[1:]}')
print(f'Rank: {t.dim()}, Shape: {t.shape}')

#3차원 tensor 만들기
t = torch.tensor([[[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9],[9, 10, 11, 12]]])
print(f'Tensor values: \n{t}')
print(f'Rank: {t.dim()}, Shape: {t.shape}')

# Broadcasting
# Same shape
print(torch.tensor([1, 4]) + torch.tensor([2, -1]))
# Vector + Scalar
print(torch.tensor([1, 4]) + torch.tensor([3]))
# 3 -> [3, 3,]
# 2 x 1 vector + 1 x 2 vector
print(torch.tensor([1, 2]) + torch.tensor([[3], [4]]))

# Matrix multiplication vs. Elementwise multiplication
m1 = torch.tensor([[1, 2], [3, 4]]) # 2 x 2
m2 = torch.tensor([[1], [2]]) # 2 x 1
print(m1.shape, m2.shape)
print(f'Matrix multiplication: \n{m1.matmul(m2)} \n{m1 @ m2}')
print(f'Elementwise multiplication: \n{m1.mul(m2)} \n{m1 * m2}')

# Mean and Standard deviation # Cannot be calculated on integers
t = torch.tensor([[1., 2.], [3., 4.]])
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.std())

# Summation
t = torch.tensor([[1., 2.], [3., 4.]])
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))

# Max and Argmax
t = torch.tensor([[1., 2.], [3., 4.]])
print(t.max())
# With specified dimension, the first is the maximum
# and the second is the argmax.
print(f'Max: {t.max(dim=0)[0]}, Argmax: {t.max(dim=0)[1]}')
print(f'Max: {t.max(dim=1)[0]}, Argmax: {t.max(dim=1)[1]}')

print(f'Max: {t.max(dim=0)}, Argmax: {t.max(dim=0)}')
print(f'Max: {t.max(dim=1)}, Argmax: {t.max(dim=1)}')

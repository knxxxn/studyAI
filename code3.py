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
print(t.squeeze())
print(t.squeeze().shape)

# Concatenation
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

# Ones and Zeros
t = torch.tensor([[3, 1, -2], [5, 0, 4]])
print(t.shape)
print(torch.ones_like(t))
print(torch.zeros_like(t))

# In-place Operation
t = torch.tensor([[1., 2.], [3., 4.]])
print(t.mul(2.))
print(t)
print(t.mul_(2.))
print(t)


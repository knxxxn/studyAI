import numpy as np

scalar = np.array(0)
print(f'Scalar value: {scalar}')
print(f'Rank: {scalar.ndim}, Shape: {scalar.shape}')

vector = np.array([0, 1, 2, 3, 4, 5, 6])
print(f'Vector values: {vector}')
print(f'Rank: {vector.ndim}, Shape: {vector.shape}')

# (batch, data)
matrix = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(f'Matrix values: \n{matrix}')
print(f'Rank: {matrix.ndim}, Shape: {matrix.shape}')

# (batch, width, height)
tensor = np.array([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
print(f'Tensor values: \n{tensor}')
print(f'Rank: {tensor.ndim}, Shape: {tensor.shape}')

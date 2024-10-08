import numpy as np

A = np.ones((2, 3))
print(A)
print(A.shape)
B = np.array([np.ones(A.shape), np.ones(A.shape)])
print(B)
print(B.shape)

num_rows, num_cols = B.shape[-2], B.shape[-1]
print(num_rows)
print(num_cols)
for m in range(2):
    print(m)

import numpy as np

a = np.array([1,2,3], dtype=float)
b = np.array([4,5,6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1

print(W)

x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")

# Rank, Projection and QR

A = np.array([[1,2], [2,4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

import torch 

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

sim = torch.dot(x, y)
sim.backward()


print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {sim.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
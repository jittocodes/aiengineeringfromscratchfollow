import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x + 1
z = y**3 + 6 * y**2 - 10 * y + 5

print(f"Y = {y}")
print(f"Z = {z}")

z.backward()

print(f"Grad: dy/dx {x.grad}")

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
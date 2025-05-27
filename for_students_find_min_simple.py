import numpy as np
import matplotlib.pyplot as plt

epochs = 50
learning_rate = 0.1
x_current = 0

history = []

for i in range(epochs):
    grad = 4 * x_current + 8
    x_current = x_current - learning_rate * grad
    history.append(x_current)

print(f"Minimum at x ≈ {x_current:.4f}")

ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
plt.xlim([-4, 4])
plt.ylim([-10, 10])
plt.grid()
x = np.arange(-4.0, 3.0, 0.1)
y = 2 * x ** 2 + 8 * x + 3
y_tag = 4 * x + 8
plt.plot(x, y, color="b", label="y")
plt.plot(x, y_tag, color="r", label="y'")
plt.scatter(history, 2 * np.array(history) ** 2 + 8 * np.array(history) + 3, color='g', s=20, label="GD steps")
plt.legend()
plt.show()
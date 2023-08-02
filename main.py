import numpy as np
import matplotlib.pyplot as plt

A = np.array([[5, -2], [2, 2]])
B = np.array([[3, 0], [0, 1]])
x0 = np.array([0, 0])

# Часовий інтервал
dt = 0.01
t = np.arange(0, 10, dt)

# Матриця збереження стану системи
x = np.zeros((len(t), 2))
x[0] = x0

# Керування
c1 = 3
c2 = 1
u = np.array([[c1], [c2]])

# Обчислення стану системи
for i in range(1, len(t)):
    x[i] = x[i-1] + dt * (np.dot(A, x[i-1]) + np.dot(B, u))

# Графік стану системи
plt.plot(t, x[:, 0], label='x1')
plt.plot(t, x[:, 1], label='x2')
plt.xlabel('Час')
plt.ylabel('Значення')
plt.title('Графік стану системи')
plt.legend()
plt.grid(True)
plt.show()
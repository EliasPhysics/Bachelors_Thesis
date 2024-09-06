import numpy as np
import matplotlib.pyplot as plt

def piecewise_linear(x, n):
    breakpoints = np.linspace(0, 1, n+1)
    y = np.zeros_like(x)
    for i in range(n):
        mask = (x >= breakpoints[i]) & (x <= breakpoints[i+1])
        x_i = (x[mask] - breakpoints[i]) / (breakpoints[i+1] - breakpoints[i])
        y[mask] = breakpoints[i]**2 + 2*breakpoints[i]*(breakpoints[i+1] - breakpoints[i])*x_i
    return y

x = np.linspace(0, 1, 1000)

plt.figure(figsize=(8, 6))
plt.plot(x, x**2, 'b-', linewidth=2, label='f0')
plt.plot(x, piecewise_linear(x, 1), 'r-', linewidth=2, label='f1')
plt.plot(x, piecewise_linear(x, 2), 'g-', linewidth=2, label='f2')
plt.plot(x, piecewise_linear(x, 4), 'k--', linewidth=2, label='f')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise Linear Approximations of x^2')
plt.legend()
plt.grid(True)
plt.show()
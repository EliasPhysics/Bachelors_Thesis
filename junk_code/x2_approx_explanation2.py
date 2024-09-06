import numpy as np
import matplotlib.pyplot as plt

def piecewise_linear_approx(x, n):
    """
    Create a piecewise linear approximation of x^2 that is always >= x^2
    and continuous, with n segments.
    """
    breakpoints = np.linspace(0, 1, n+1)
    y = np.zeros_like(x)
    for i in range(n):
        mask = (x >= breakpoints[i]) & (x <= breakpoints[i+1])
        x0, x1 = breakpoints[i], breakpoints[i+1]
        y0, y1 = x0**2, x1**2
        slope = (y1 - y0) / (x1 - x0)
        y[mask] = y0 + slope * (x[mask] - x0)
    return y

x = np.linspace(0, 1, 1000)

plt.figure(figsize=(8, 6))
plt.plot(x, x**2, 'b-', linewidth=2, label='f0')
plt.plot(x, piecewise_linear_approx(x, 1), 'r-', linewidth=2, label='f1')
plt.plot(x, piecewise_linear_approx(x, 2), 'g-', linewidth=2, label='f2')
plt.plot(x, piecewise_linear_approx(x, 4), 'k--', linewidth=2, label='f')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise Linear Approximations of x^2')
plt.legend()
plt.grid(True)
plt.show()
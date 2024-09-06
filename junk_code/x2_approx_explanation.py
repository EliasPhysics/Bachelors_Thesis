import os
import numpy as np
import matplotlib.pyplot as plt

def g(x):
    """
    Defines the custom "tooth" (mirror) function g: [0, 1] -> [0, 1] with specific behavior:
    g(x) = 2x for x < 1/2,
    g(x) = 2(1-x) for x >= 1/2.
    """
    return np.where(x < 0.5, 2 * x, 2 * (1 - x))

def g_n(x, n):
    """
    Applies the g function n times to create a sawtooth with n teeth.
    """
    y = x.copy()
    y = g(n * y % 1)
    return y

x = np.linspace(0, 1, 1000)

plt.figure(figsize=(8, 6))
plt.plot(x, g_n(x, 1), 'r-', linewidth=2, label='g')
plt.plot(x, g_n(x, 2), 'g-', linewidth=2, label="g2")
plt.plot(x, g_n(x, 4), 'y-', linewidth=2, label='g3')

plt.xlim(0, 1)
plt.ylim(0, 1)
#plt.tight_layout()
plt.legend()
plt.tight_layout()
os.chdir("..")
plt.savefig("Plots/x2_approx_explanation_sawtooths.pdf")
plt.show()
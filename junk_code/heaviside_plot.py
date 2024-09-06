import numpy as np
import matplotlib.pyplot as plt

def rho(x):
    return np.maximum(x, 0)

def approximated_heaviside(x, epsilon):
    return rho(x/epsilon) - rho(x/epsilon - 1)

# Set up the plot
epsilon = 0.1
x = np.linspace(-0.5, 0.5, 1000)
y = approximated_heaviside(x, epsilon)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'Approximation (ε = {epsilon})')

# Plot the true Heaviside function
y_true = np.heaviside(x, 0.5)
plt.plot(x, y_true, '--', label='True Heaviside')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Highlight the transition region
plt.axvspan(0, epsilon, alpha=0.2, color='red', label='Transition region')

plt.legend()
plt.savefig("../Plots/heaviside_approx_plot.pdf")
plt.show()
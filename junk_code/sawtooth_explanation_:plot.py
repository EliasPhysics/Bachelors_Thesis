import matplotlib.pyplot as plt
import numpy as np

# Define the sigmoid function
def sigma(x):
    return np.maximum(0,x)

# Define the piecewise function g(x)
def g(x):
    """
    Defines the custom "tooth" (mirror) function g: [0, 1] -> [0, 1] with specific behavior:
    g(x) = 2x for x < 1/2,
    g(x) = 2(1-x) for x >= 1/2.

    Parameters:
    - x: A single value or NumPy array of x values in the range [0, 1].

    Returns:
    - The evaluated result(s) according to the piecewise function definition.
    """
    return np.where(x < 0.5, 2 * x, 2 * (1 - x))

# Create x values
x = np.linspace(0, 1.2, 400)

# Calculate y values for each function
y_g = g(x)
y_2sigma = 2 * sigma(x)
y_neg4sigma = -4 * sigma(x - 0.5)
y_2sigma_shifted = 2 * sigma(x - 1)

# Create the plot
plt.plot(x, y_g, label='g(x)', color='blue')
plt.plot(x, y_2sigma, label='2*$\sigma$(x)', linestyle='--', color='red')
plt.plot(x, y_neg4sigma, label='-4*$\sigma$(x - 0.5)', linestyle='--', color='green')
plt.plot(x, y_2sigma_shifted, label='2*$\sigma$(x - 1)', linestyle='--', color='orange')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ylim(-1.5, 2)
plt.xlim(0,1)
plt.savefig("Plots/sawtooth_explanation_plot.pdf")
plt.show()

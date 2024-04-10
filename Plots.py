
import matplotlib.pyplot as plt
import numpy as np


def custom_tooth_function(x):
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


# Generate a range of x values within [0, 1]
x_values = np.linspace(0, 1, 500)  # Generating 500 points for smoothness

# Apply the custom tooth function to generate y values
y_values = custom_tooth_function(x_values)

# Plotting the custom tooth function
plt.figure(figsize=(6, 6))
plt.plot(x_values, y_values, label='Zahn Funktion')
plt.xlabel('x')
plt.ylim(0, 1)  # Setting the y-axis limits to match the function's range
plt.legend()
plt.grid(True)
plt.savefig("Tooth_Plot")


def double_tooth_function(x):
    """
    Adjusts the custom "tooth" function to feature two peaks or "teeth" within the interval [0, 1].

    Parameters:
    - x: A single value or NumPy array of x values in the range [0, 1].

    Returns:
    - The evaluated result(s) according to the adjusted piecewise function definition.
    """
    # Scale the x-values to fit two cycles within [0, 1]
    scaled_x = 8 * x  # This scales the pattern to repeat twice within [0, 1]

    # Apply the original logic, but on the scaled x-values
    return np.where(scaled_x % 2 < 1,  (scaled_x % 1),  (1 - (scaled_x % 1)))


# Generate a range of x values within [0, 1]
x_values = np.linspace(0, 1, 1000)  # Generating 1000 points for higher resolution

# Apply the double tooth function to generate y values
y_values = double_tooth_function(x_values)

# Plotting the double tooth function
plt.figure(figsize=(6, 6))
plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylim(0, 1)  # Ensuring y-axis limits match the function's range
plt.legend()
plt.grid(True)
plt.savefig("Quad_tooth_Plot")
# The function calls in the block are commented to adhere to the policy. They will be uncommented in the final code.

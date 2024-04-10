import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the function
def f(x):
    return x ** 2

# Interval [a, b]
a, b = 0, 2

# Calculate the total area under the curve using the integral of f(x) = x^2 from a to b
total_area = (b ** 3 - a ** 3) / 3

# Fraction of the area to be represented by the square
fraction = 1/4

# Area of the square
square_area = total_area * fraction

# Side length of the square (area of a square = side^2)
side_length = 1
step_x = [a]
step_y = [f(a)]
# Choose a position for the square (for simplicity, start at x=1 so it fits under the curve nicely)
square_x = 1
square_y = 0  # Start on the x-axis

# Plot the function
x = np.linspace(a, b, 400)
y = f(x)
plt.plot(x, y, label='f(x) = $x^2$')

# Add a square under the curve
plt.gca().add_patch(Rectangle((square_x, square_y), side_length, side_length, linewidth=1, edgecolor='r', facecolor='none'))

step_x.extend([1, 2])
step_y.extend([1, 1])
plt.step(step_x, step_y, where='post', label='Step Function Approximation', color='black', zorder=3)

# Adjusting plot
plt.xlim([a, b])
plt.ylim([0, max(y)])
plt.xlabel('x')
plt.grid(True)
plt.savefig("Approx_Onesquare")
plt.close()

step_x = [a]
step_y = [f(a)]
x = np.linspace(a, b, 400)
y = f(x)
plt.plot(x, y, label='f(x) = $x^2$')
# Add a square under the curve
plt.gca().add_patch(Rectangle((square_x, square_y), side_length, side_length, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((1.5, 1),2,  1.5**2-1, linewidth=1, edgecolor='r', facecolor='none'))

step_x.extend([1, 1.5])
step_y.extend([1, 1])
step_x.extend([1.5, 2])
step_y.extend([1.5**2, 1.5**2])

# Plot the step function to approximate the original function
plt.step(step_x, step_y, where='post', label='Step Function Approximation', color='black', zorder=3)


# Adjusting plot
plt.xlim([a, b])
plt.ylim([0, max(y)])
plt.xlabel('x')
plt.grid(True)
plt.savefig("Approx_twosquare")
plt.close()

# Prepare the plot of the function again and the rectangles
plt.plot(x, y, zorder=1)  # Original function

# Store x and y values for the step function
step_x = [a]
step_y = [f(a)]

n = 10
interval_width = 2/n

# Plot each square using the left endpoint for height and create the step function
for i in range(n):
    x_left = a + interval_width * i
    height = f(x_left)

    # Coordinates for the squares
    square_x = x_left
    square_y = 0

    # Plot the square
    plt.gca().add_patch(
        Rectangle((square_x, square_y), interval_width, height, linewidth=1, edgecolor='r', facecolor='none', zorder=2))

    # Add points for the step function
    step_x.extend([x_left, x_left + interval_width])
    step_y.extend([height, height])

# Plot the step function to approximate the original function
plt.step(step_x, step_y, where='post', label='Step Function Approximation', color='black', zorder=3)

# Adjusting plot
plt.xlim([a, b])
plt.ylim([0, max(y)])
plt.xlabel('x')
plt.grid(True)
plt.savefig("Approx_manysquares")

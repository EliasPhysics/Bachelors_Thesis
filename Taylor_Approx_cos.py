import numpy as np
import matplotlib.pyplot as plt

# Define the Taylor series approximation of cos(x)
def taylor_cos(x, n):
    approximation = np.zeros_like(x)
    for k in range(n + 1):
        term = ((-1) ** k) * (x ** (2 * k)) / np.math.factorial(2 * k)
        approximation += term
    return approximation

# Define the x range
x = np.linspace(-np.pi, np.pi, 400)

# Calculate the exact cos(x)
y_cos = np.cos(x)

# Calculate Taylor approximations
y_taylor_2 = taylor_cos(x, 1)  # 2nd order
y_taylor_4 = taylor_cos(x, 2)  # 4th order
y_taylor_6 = taylor_cos(x, 3)  # 6th order
y_taylor_8 = taylor_cos(x, 4)  # 8th order

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the exact cos(x)
plt.plot(x, y_cos, label='cos(x)', color='blue')

# Plot Taylor approximations
plt.plot(x, y_taylor_2, label='2nd order: $1 - \\frac{x^2}{2!}$', linestyle='--', color='red')
plt.plot(x, y_taylor_4, label='4th order: $1 - \\frac{x^2}{2!} + \\frac{x^4}{4!}$', linestyle='--', color='green')
plt.plot(x, y_taylor_6, label='6th order: $1 - \\frac{x^2}{2!} + \\frac{x^4}{4!} - \\frac{x^6}{6!}$', linestyle='--', color='orange')
plt.plot(x, y_taylor_8, label='8th order: $1 - \\frac{x^2}{2!} + \\frac{x^4}{4!} - \\frac{x^6}{6!} + \\frac{x^8}{8!}$', linestyle='--', color='purple')

# Customize the plot
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.ylim([-1.1,1.1])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
plt.yticks([-1, 0, 1])

# Show the plot
plt.savefig('Taylor_Approximation_cos.pdf')
plt.show()

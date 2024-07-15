# Adjusting the approximation function for 6 intervals with Taylor expansions of degree 2
import numpy as np
import matplotlib.pyplot as plt

def part_unity(x):
    x = np.abs(x)
    result = np.zeros_like(x)
    result[x < 1/4] = 1
    mask = (x >= 1/4) & (x < 2/4)
    result[mask] = 2 - 4 * x[mask]
    return result

def approximate_sin_2nd_degree(x):
    """
    Approximates sin(6*pi*x) using 6 local Taylor expansions of degree 2.

    Parameters:
        x (float): The input value to approximate sin for.

    Returns:
        float: The approximated value of sin(6*pi*x).
    """
    # Scaling x for sin(6*pi*x)
    x_scaled = 6 * np.pi * x

    # Determine the interval and corresponding center point for Taylor expansion
    # There are 6 intervals, each centered around multiples of π/3 within the 6π range
    centers = np.arange(0, 1)
    a = centers[np.searchsorted(centers, x_scaled, side='right') - 1]

    # Determine sin(a) and cos(a) based on the center a
    sin_a = np.sin(a)
    cos_a = np.cos(a)

    # Calculate the second-degree Taylor series approximation
    x_diff = x_scaled - a
    approximation = sin_a + x_diff * cos_a - (x_diff ** 2) / np.factorial(2) * sin_a

    return approximation

def approx_sin(x):
    list_a = np.arange(1/2,6,1)
    pi = np.pi
    approximation = 0
    for a in list_a:
        approximation += part_unity(x - a)*( np.sin(a*pi) + (x-a) * np.cos(a*pi) *pi - ((x-a) ** 2) * np.sin(a*pi) * pi**2 / 2)
    return approximation

def plot_taylor_approx():
    x_values = np.linspace(0,6,500)
    y_values_approx = [approx_sin(x) for x in x_values]
    y_values = [np.sin(np.pi*x) for x in x_values]
    y_values_unity = [part_unity(x) for x in x_values]

    # Generate approximations using the new approximate_sin_2nd_degree function

    # Re-plotting sin(6*pi*x) with the new approximation
    plt.figure(figsize=(10, 6))  # Making the figure larger for better visibility
    plt.plot(x_values, y_values, label='sin(πx)', linestyle='-')
    plt.plot(x_values, y_values_approx, label='Approximation (2nd Degree)', linestyle='--', color='red')
    for i in range(7):
        plt.vlines([i], [-1], [1],linestyle='--', color='gray')
    plt.xlabel('x')
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig("local_taylor_approx_plot.pdf")
    plt.show()

def plot_part_unity():
    x = np.linspace(0, 9 / 10 + 1 / 5,500)
    x1 = np.linspace(0,1/5,100)
    x2= np.linspace(3/10-1/5,3/10+1/5,100)
    x3 = np.linspace(6 / 10 - 1 / 5, 6 / 10 + 1 / 5, 100)
    x4 = np.linspace(9 / 10 - 1 / 5, 9 / 10 + 1 / 5, 100)
    y = part_unity(x1)
    y2 = part_unity(x2-3/10)
    y3 = part_unity(x3-6/10)
    y4 = part_unity(x4-9/10)

    plt.plot(x1, y, label='')
    plt.plot(x2, y2)
    plt.plot(x3, y3)
    plt.plot(x4, y4)
    plt.plot(x, np.ones(500),c="g" , linestyle='--')
    plt.tight_layout()
    plt.savefig("partition_of_unity.pdf")
    plt.show()


if __name__ == "__main__":
    #plot_part_unity()
    plot_taylor_approx()
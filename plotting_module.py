import matplotlib.pyplot as plt
import numpy as np
from approx import f1,f2,f3



def plot_accuracy(f):
    mean_list = np.load(f"Data_pc/variable_layers_means_{f.__name__}.npy")
    std_list = np.load(f"Data_pc/variable_layers_std_{f.__name__}.npy")
    widths = [5 + i for i in range(len(mean_list))]
    plt.errorbar(widths, mean_list, yerr=std_list/2, fmt='o', capsize=5, capthick=2, ecolor='red', linestyle='None')
    plt.title(f"Accuracy for {f.__name__}")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"Plots/variable_widths_{f.__name__}.svg")
    plt.show()


if __name__ == '__main__':
    function_list = [f1,f2,f3]
    for f in function_list:
        plot_accuracy(f)

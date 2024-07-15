import os

import matplotlib.pyplot as plt
import numpy as np
from bin.scaling_laws import f1,f2,f3



def plot_accuracy(f):
    mean_list = np.load(f"Data_pc_new/variable_widths_means_{f.__name__}.npy")
    std_list = np.load(f"Data_pc_new/variable_widths_std_{f.__name__}.npy")
    widths = [1 + i for i in range(len(mean_list))]
    plt.errorbar(widths, mean_list, yerr=std_list/2, fmt='o', capsize=5, capthick=2, ecolor='red', linestyle='None')
    #plt.title(f"Accuracy for {f.__name__}")
    plt.yscale("log")
    plt.ylabel("approximation error " + r'$\epsilon$')
    plt.xlabel("Neural Network width")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"Plots/variable_widths_{f.__name__}.pdf")
    plt.show()


if __name__ == '__main__':
    os.chdir("..")
    function_list = [f1,f2,f3]
    for f in function_list:
        plot_accuracy(f)

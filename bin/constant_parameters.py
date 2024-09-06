import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import re


# Generate dataset

def model(layers,width,f,plot = False):
    x_train = np.linspace(0, 1, 10000)
    y_train = f(x_train)
    # Neural network model
    model = Sequential()

    # Input layer
    model.add(Dense(width, input_shape=(1,), activation='relu'))

    # Hidden layers
    for _ in range(layers-1):
        model.add(Dense(width, activation='relu'))

    # Output layer
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Train model
    model.fit(x_train, y_train, epochs=150, verbose=0)

    # Generate test data
    x_test = np.linspace(0, 1, 10000)
    y_test = f(x_test)
    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()
    max_error = np.max(abs(y_test - y_pred))
    print(f"model layers x width: {layers}x{width}  max: {max_error}")
    error = np.sqrt(np.sum((y_test - y_pred) ** 2) / len(y_test))  # Step 1: Calculate differences
    print(f"model layers x width: {layers}x{width} loss:{error}")

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x_test, y_test)
        ax.plot(x_test, y_pred, label='Approximation', linestyle='--')

        # Add an inset - zoom into the range [-2, 2] on x and [0, 10] on y
        ax_inset = inset_axes(ax, width="30%", height="30%", loc=3)  # loc=3 is lower left corner
        ax_inset.plot(x_test, y_test, label='Zoom f(x) = sin(x)')
        ax_inset.plot(x_test, y_pred, )
        ax_inset.set_xlim(0.1, 0.4)  # Limits for x-axis
        ax_inset.set_ylim(0.9, 1.1)  # Limits for y-axis
        ax_inset.set_title('Zoom')

        # Optionally add marks that connect the area of the zoom to the larger plot
        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        # Show the plot
        plt.savefig("Plots/approx_sin_with_zoom.pdf")
        plt.show()
        tf.keras.backend.clear_session()
    return error,max_error

def test_model(layers,width,f,n=1):
    error_list = []
    max_error_list = []
    for i in range(n):
        error, max_error = model(layers=layers, width=width, f=f)
        error_list.append(error)
        max_error_list.append(max_error)
    error_list = np.array(error_list)
    np.save(f"Data/error_list_{layers}x{width}_model_{f.__name__}.npy", error_list)
    np.save(f"Data/max_error_list_{layers}x{width}_model_{f.__name__}.npy",max_error_list)
    return error_list, max_error_list

def f1(x):
    return np.sin(2*np.pi*x)
def f2(x):
    return x*x

def f3(x):
    return np.heaviside(x-1/2, 1)

def _extract_first_number(string):
    match = re.search(r'\d+', string)
    print(int(match.group()) if match else "No Number found")
    return int(match.group()) if match else None


def _sort_lists(list1, list2, std = None , min = None):
    # Create a list of tuples, where each tuple contains the index and value from list1
    indexed_list1 = list(enumerate(list1))

    # Sort the indexed list based on the values (second element of each tuple)
    sorted_indexed_list1 = sorted(indexed_list1, key=lambda x: x[1])

    # Extract the sorted indices
    sorted_indices = [index for index, value in sorted_indexed_list1]


    # Use the sorted indices to reorder both lists
    sorted_list1 = [list1[i] for i in sorted_indices]
    sorted_list2 = [list2[i] for i in sorted_indices]
    std_sorted = [std[i] for i in sorted_indices]
    min_sorted = [min[i] for i in sorted_indices]
    return sorted_list1, sorted_list2, std_sorted, min_sorted


import matplotlib.pyplot as plt


def eval_results(dir, functions):
    files = os.listdir(dir)
    max_error_list_files = [f for f in files if "max_error_list" in f]
    error_list_files = [f for f in files if "error_list" in f and not "max_error_list" in f]

    for func in functions:
        max_error_list_files_f = [f for f in max_error_list_files if f"{func.__name__}" in f]
        error_list_files_f = [f for f in error_list_files if f"{func.__name__}" in f]
        error_data_f_mean = []
        max_error_data_f_mean = []

        max_error_data_f_min = []
        max_error_data_f_std = []

        error_data_f_min = []
        error_data_f_std = []
        n_layers = []
        n_layers_max = []

        for file in max_error_list_files_f:
            data = np.load("Data/" + file)
            max_error_data_f_min.append(min(data))
            max_error_data_f_mean.append(np.mean(data))
            max_error_data_f_std.append(np.std(data))
            layers = _extract_first_number(file)
            n_layers_max.append(layers)

        layers_sorted_max, max_error_sorted, max_error_std_sorted, max_error_min_sorted= _sort_lists(n_layers_max, max_error_data_f_mean, std = max_error_data_f_std, min =  max_error_data_f_min)


        for file in error_list_files_f:
            data = np.load("Data/" + file)
            error_data_f_min.append(min(data))
            error_data_f_mean.append(np.mean(data))
            error_data_f_std.append(np.std(data))
            layers = _extract_first_number(file)
            n_layers.append(layers)

        layers_sorted, error_sorted,error_std_sorted,error_min_sorted = _sort_lists(n_layers, error_data_f_mean,min=error_data_f_min,std=error_data_f_std)

        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot L2 error
        ax1.plot(layers_sorted_max, max_error_min_sorted)
        #ax1.errorbar(layers_sorted, max_error_sorted , yerr=np.array(max_error_std_sorted) / 2, fmt='o', capsize=5, capthick=2, ecolor='red',linestyle='None', label="mean")
        ax1.set_title(r"$L_2$-error for " + f"{func.__name__}")
        ax1.grid()
        ax1.legend()
        # ax1.set_yscale("log")  # Uncomment if you want log scale

        # Plot L-infinity error
        ax2.plot(layers_sorted, error_min_sorted)
        #ax2.errorbar(layers_sorted, error_sorted, yerr=np.array(error_std_sorted) / 2, fmt='o', capsize=5, capthick=2,ecolor='red',linestyle='None', label="mean")
        ax2.set_title(r"$L_\infty$-error for " + f"{func.__name__}")
        ax2.legend()
        # ax2.set_yscale("log")  # Uncomment if you want log scale

        # Adjust layout and save figure
        plt.tight_layout()
        ax2.grid()
        plt.savefig(f"Plots/error_constant_params_{func.__name__}_plot.pdf")
        #plt.close()
        plt.show()

functions = [f1,f2,f3]

def data_generation_run():
    #model(layers=5, width=5, f=f1, plot=True)
    for f in functions:
       # test_model(layers=12, width=3, f=f, n=50)
      #  test_model(layers=7, width=4, f=f, n=30)
       # test_model(layers=5, width=5, f=f, n=15)
        #test_model(layers=3, width=7, f=f, n=15)
       # test_model(layers=2, width=10, f=f, n=15)
        test_model(layers=1, width=45, f=f, n=15)

if __name__ == "__main__":
    os.chdir("..")
    data_generation_run()
    eval_results(dir="Data",functions=functions)

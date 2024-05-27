import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from tqdm import tqdm

def approximate(f,layers,width):
    x_train = np.linspace(0, 1, 1000)
    y_train = f(x_train)
    # Neural network model
    model = Sequential()

    # Input layer
    model.add(Dense(width, input_shape=(1,), activation='relu'))

    # Hidden layers
    for _ in range(layers - 1):
        model.add(Dense(width, activation='relu'))

    # Output layer
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(x_train, y_train, epochs=150, verbose=0)

    # Generate test data
    x_test = np.linspace(0, 1, 100)
    y_test = f(x_test)
    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()
    #model.summary()
    #print(f"max: {np.max(abs(y_test-y_pred))}")
    error = np.sqrt(np.sum((y_test - y_pred)**2)/100)  # Step 1: Calculate differences
    tf.keras.backend.clear_session()
    return error


def test_approx(n,f,layers,width):
    error_list = []
    for i in range(n):
        error = approximate(f,layers=layers,width=width)
        error_list.append(error)
    error_list = np.array(error_list)
    return error_list, np.mean(error_list), np.std(error_list)


def f1(x):
    return np.sin(2*np.pi*x)
def f2(x):
    return x*x

def f3(x):
    return np.heaviside(x-1/2, 1)

if __name__ == '__main__':
    functions = [f1,f2,f3]
    widths = [5 + i for i in range(30)]
    for f in functions:
        mean_list = np.array([])
        std_list = np.array([])
        for w in tqdm(widths):
            error_list, mean, std = test_approx(n=10, f=f, layers=5, width=w)
            mean_list = np.append(mean_list, mean)
            std_list = np.append(std_list, std)
            np.save(f"Data/variable_layers_means_{f.__name__}", mean_list)
            np.save(f"Data/variable_layers_std_{f.__name__}", std_list)
        plt.errorbar(widths, mean_list, yerr=std_list, fmt='o', capsize=5, capthick=2, ecolor='red', linestyle='None')
        plt.savefig(f"variable_widths_{f.__name__}.svg")
        plt.show()

    layers = [5 + i for i in range(30)]
    for f in functions:
        mean_list = np.array([])
        std_list = np.array([])
        for l in tqdm(layers):
            error_list, mean, std = test_approx(n=10, f=f, layers=l, width=5)
            mean_list = np.append(mean_list,mean)
            std_list = np.append(std_list,std)
            np.save(f"Data/variable_layers_means_{f.__name__}", mean_list)
            np.save(f"Data/variable_layers_std_{f.__name__}", std_list)
        plt.errorbar(layers, mean_list, yerr=std_list, fmt='o', capsize=5, capthick=2, ecolor='red', linestyle='None')
        plt.savefig(f"variable_layers_{f.__name__}.svg")
        plt.show()

 # std und theorieschranken einbauen
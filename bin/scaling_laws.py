import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras


def create_dense_network(num_layers, width):
    inputs = keras.Input(shape=(1,))
    x = inputs

    # List to store outputs of each layer for skip connections
    layer_outputs = []

    for i in range(num_layers):
        # Concatenate all previous outputs with the current input
        if i > 0:
            x = keras.layers.Concatenate()([x] + layer_outputs)

        # Create a dense layer
        dense = keras.layers.Dense(width, activation='relu')(x)

        # Add the output to our list of layer outputs
        layer_outputs.append(dense)

        # Update x for the next iteration
        x = dense

    # Final concatenation of all layer outputs
    x = keras.layers.Concatenate()([x] + layer_outputs)

    # Output layer
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model





def approximate(f,layers,width):
    X = np.linspace(0, 1, 10000)
    Y = f(X)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    # Neural network model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                             patience=15)
    model = create_dense_network(layers, width)


    model.compile(tf.keras.optimizers.Adam(), loss='mse')

    # Train model
    model.fit(x_train, y_train, epochs=150, verbose=0,callbacks=[callback])

    # Generate test data

    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()
    #model.summary()
    #print(f"max: {np.max(abs(y_test-y_pred))}")
    error = np.sqrt(np.sum((y_test - y_pred)**2)/len(x_test))  # Step 1: Calculate differences
    tf.keras.backend.clear_session()
    return error


def test_approx(n,f,layers,width):
    error_list = []
    for i in range(n):
        error = approximate(f,layers=layers,width=width)
        error_list.append(error)
    error_list = np.array(error_list)
    return error_list, np.mean(error_list), np.std(error_list)


def plot_scaling_laws(f, layers = None, widths = None):
   # means = np.load(f"Data/variable_layers_means_{f.__name__}.npy")
   # std = np.load(f"Data/variable_layers_std_{f.__name__}.npy")
    if layers is not None:
        min = np.load(f"Data/variable_layers_min_{f.__name__}.npy")
        #plt.errorbar(layers, mean_list, yerr=std_list / 2, fmt='o', capsize=5, capthick=2, ecolor='red',
             #        linestyle='None')
        plt.plot(layers, min,linestyle="-", label="min")
        plt.yscale("log")
        plt.title(f"Approximation error for different depths when approximating {f.__name__}")
        plt.ylabel("error")
        plt.xlabel("depth")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Plots/variable_layers_{f.__name__}_min.pdf")
        plt.show()
    elif widths is not None:
        min = np.load(f"Data/variable_widths_min_{f.__name__}.npy")
        plt.plot(widths, min, linestyle="-", label="min")
        plt.yscale("log")
        plt.title(f"Approximation error for different widths when approximating {f.__name__}")
        plt.ylabel("error")
        plt.xlabel("width")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Plots/variable_widths_{f.__name__}_min.pdf")
        plt.show()


def f1(x):
    return np.sin(2*np.pi*x)
def f2(x):
    return x*x

def f3(x):
    return np.heaviside(x-1/2, 1)

if __name__ == '__main__':
    os.chdir("..")
    functions = [f1,f2,f3]
    widths = [1 + i for i in range(30)]
    for f in functions:
        #mean_list = np.array([])
        #std_list = np.array([])
        #min_list = np.array([])
        #for w in tqdm(widths):
        #    error_list, mean, std = test_approx(n=10, f=f, layers=2, width=w)
        #    mean_list = np.append(mean_list, mean)
        #    std_list = np.append(std_list, std)
        #    min_list = np.append(min_list, min(error_list))
        #    np.save(f"Data/variable_widths_means_{f.__name__}", mean_list)
        #    np.save(f"Data/variable_widths_std_{f.__name__}", std_list)
        #    np.save(f"Data/variable_widths_min_{f.__name__}.npy", min_list)
        #    tf.keras.backend.clear_session()
        #plt.errorbar(widths, mean_list, yerr=std_list/2, fmt='o', capsize=5, capthick=2, ecolor='red', linestyle='None')
        #plt.yscale("log")
        #plt.savefig(f"Plots/variable_widths_{f.__name__}.pdf")
        #plt.show()
        #plt.close()
        plot_scaling_laws(f, widths=widths)

    layers = [1 + i for i in range(15)]
    for f in functions:
        plot_scaling_laws(f,layers=layers)
     #   mean_list = np.array([])
     #   std_list = np.array([])
     #   min_list = np.array([])
     #   for l in tqdm(layers):
     #       error_list, mean, std = test_approx(n=15, f=f, layers=l, width=4)
     #       mean_list = np.append(mean_list,mean)
     #       std_list = np.append(std_list,std)
     #       min_list = np.append(min_list,min(error_list))
     #       np.save(f"Data/variable_layers_means_{f.__name__}.npy", mean_list)
     #       np.save(f"Data/variable_layers_std_{f.__name__}.npy", std_list)
     #       np.save(f"Data/variable_layers_min_{f.__name__}.npy", min_list)
     #       tf.keras.backend.clear_session()
     #   plt.errorbar(layers, mean_list, yerr=std_list/2, fmt='o', capsize=5, capthick=2, ecolor='red', linestyle='None')
     #   plt.plot(layers, min_list, linestyle = "--", label="min")
     #   plt.yscale("log")
     #   plt.savefig(f"variable_layers_{f.__name__}.pdf")
     #   plt.show()




 # std und theorieschranken einbauen
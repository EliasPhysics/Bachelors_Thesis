import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd


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
    max_error = np.max(abs(y_test - y_pred))
    error = np.sqrt(np.sum((y_test - y_pred)**2)/len(x_test))  # Step 1: Calculate differences
    tf.keras.backend.clear_session()
    return error, max_error


def test_approx(n,f,layers,width):
    error_list = []
    max_error_list = []
    for i in range(n):
        error, max_error = approximate(f,layers=layers,width=width)
        error_list.append(error)
        max_error_list.append(max_error)
    error_list = np.array(error_list)
    return error_list, max_error_list


def plot_scaling_laws(f, layers=None, widths=None):
    if layers is not None:
        # Load data from CSV
        df = pd.read_csv(f"Data/error_data_layers_{f.__name__}.csv")

        # Group by layers and calculate statistics
        grouped = df.groupby('width').agg({
            'error_list': ['min', 'mean', lambda x: x.std() / 2]
        }).reset_index()
        grouped.columns = ['layers', 'min', 'mean', 'std']

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['layers'], grouped['min'], linestyle="-", label="min")
        plt.errorbar(grouped['layers'], grouped['mean'], yerr=grouped['std'], fmt='o', capsize=5, capthick=2,
                     ecolor='red', linestyle='None', label="mean")
        plt.yscale("log")
        plt.title(r"$L_2$-error" + f" for different depths when approximating {f.__name__}")
        plt.ylabel("error")
        plt.xlabel("depth")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Plots/variable_layers_{f.__name__}_min.pdf")
        plt.show()

    elif widths is not None:
        # Load data from CSV
        df = pd.read_csv(f"Data/error_data_width_{f.__name__}.csv")

        # Group by widths and calculate statistics
        grouped = df.groupby('width').agg({
            'error_list': ['min', 'mean', lambda x: x.std() / 2]
        }).reset_index()
        grouped.columns = ['width', 'min', 'mean', 'std']

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['width'], grouped['min'], linestyle="-", label="min")
        plt.errorbar(grouped['width'], grouped['mean'], yerr=grouped['std'], fmt='o', capsize=5, capthick=2,
                     ecolor='red', linestyle='None', label="mean")
        plt.yscale("log")
        plt.title(r"$L_2$-error" + f" for different widths when approximating {f.__name__}")
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
    functions = [f2, f3]
    widths = [1 + i for i in range(30)]

    #for f in functions:
    #    error_data = []
    #    max_error_data = []

     #   for w in tqdm(widths):
     #       error_list, max_error_list = test_approx(n=20, f=f, layers=2, width=w)

            # Append data for error DataFrame
    #        error_data.append({
    #            'width': w,
    #            'error_list': error_list,
    #            'mean': np.mean(error_list),
    #            'std': np.std(error_list),
    #            'min': np.min(error_list)
    #        })

            # Append data for max_error DataFrame
    #        max_error_data.append({
    #            'width': w,
    #            'max_error_list': max_error_list,
    #            'mean': np.mean(max_error_list),
    #            'std': np.std(max_error_list),
     #           'min': np.min(max_error_list)
     #       })

      #      tf.keras.backend.clear_session()

            # Create DataFrames
      #      error_df = pd.DataFrame(error_data)
      #      max_error_df = pd.DataFrame(max_error_data)

            # Save DataFrames to CSV
     #       error_df.to_csv(f"Data/error_data_width_{f.__name__}.csv", index=False)
     #       max_error_df.to_csv(f"Data/max_error_data_width_{f.__name__}.csv", index=False)
        #plot_scaling_laws(f,widths=widths)

    functions = [f2, f3]
    layers = [1 + i for i in range(13)]
    for f in functions:
        error_data = []
        max_error_data = []

        for l in tqdm(layers):
            error_list, max_error_list = test_approx(n=20, f=f, layers=l, width=4)

            # Append data for error DataFrame
            error_data.append({
                'depth': l,
                'error_list': error_list,
                'mean': np.mean(error_list),
                'std': np.std(error_list),
                'min': np.min(error_list)
            })

            # Append data for max_error DataFrame
            max_error_data.append({
                'depth': l,
                'max_error_list': max_error_list,
                'mean': np.mean(max_error_list),
                'std': np.std(max_error_list),
                'min': np.min(max_error_list)
            })

            tf.keras.backend.clear_session()

            # Create DataFrames
            error_df = pd.DataFrame(error_data)
            max_error_df = pd.DataFrame(max_error_data)

            # Save DataFrames to CSV
            error_df.to_csv(f"Data/error_data_layers_{f.__name__}.csv", index=False)
            max_error_df.to_csv(f"Data/max_error_data_layers_{f.__name__}.csv", index=False)
        #plot_scaling_laws(f, layers=layers)




 # std und theorieschranken einbauen
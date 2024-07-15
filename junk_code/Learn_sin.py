import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# Generate dataset


def model1():
    x_train = np.linspace(0, 2 * np.pi, 1000)
    y_train = np.sin(x_train)
    # Neural network model
    model = Sequential([
        Dense(10, input_shape=(1,), activation='relu'),
        Dense(10, input_shape=(1,), activation='relu'),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(x_train, y_train, epochs=150, verbose=0)

    # Generate test data
    x_test = np.linspace(0, 2*np.pi, 100)
    y_test = np.sin(x_test)
    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()
    model.summary()
    print(f"max: {np.max(abs(y_test-y_pred))}")
    error = np.sqrt(2*np.pi*np.sum((y_test - y_pred)**2)/100)  # Step 1: Calculate differences
    print(f"loss:{error}")


    fig, ax = plt.subplots()
    ax.plot(x_test, y_test, label='f(x) = x^2')
    ax.plot(x_test, y_pred, label='Approximation', linestyle='--')


    # Add an inset - zoom into the range [-2, 2] on x and [0, 10] on y
    ax_inset = inset_axes(ax, width="30%", height="30%", loc=3)  # loc=3 is lower left corner
    ax_inset.plot(x_test, y_test, label='Zoom f(x) = sin(x)')
    ax_inset.plot(x_test, y_pred, )
    ax_inset.set_xlim(1.5, 2)  # Limits for x-axis
    ax_inset.set_ylim(0.75, 1.25)  # Limits for y-axis
    ax_inset.set_title('Zoom')

    # Optionally add marks that connect the area of the zoom to the larger plot
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    # Show the plot
    plt.show()



def model2():
    x_train = np.linspace(0, 2 * np.pi, 1000)
    y_train = np.sin(x_train)
    # Neural network model
    model = Sequential([
        Dense(7, input_shape=(1,), activation='relu'),
        Dense(7, activation='relu'),
        Dense(7, activation='relu'),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(x_train, y_train, epochs=150, verbose=0)

    # Generate test data
    x_test = np.linspace(0, 2 * np.pi, 100)
    y_test = np.sin(x_test)
    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()
    model.summary()
    print(f"model 2 max: {np.max(abs(y_test - y_pred))}")
    error = np.sqrt(2 * np.pi * np.sum((y_test - y_pred) ** 2) / 100)  # Step 1: Calculate differences
    print(f"model 2 loss:{error}")

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test)
    ax.plot(x_test, y_pred, linestyle='--')

    ax.legend()

    # Add an inset - zoom into the range [-2, 2] on x and [0, 10] on y
    ax_inset = inset_axes(ax, width="30%", height="30%", loc=3)  # loc=3 is lower left corner
    ax_inset.plot(x_test, y_test)
    ax_inset.plot(x_test, y_pred, linestyle='--')
    ax_inset.set_xlim(1.5, 2)  # Limits for x-axis
    ax_inset.set_ylim(0.75, 1.25)  # Limits for y-axis
    ax_inset.set_title('Zoom')

    # Optionally add marks that connect the area of the zoom to the larger plot
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    # Show the plot
    plt.show()



def model3():
    x_train = np.linspace(0, 2 * np.pi, 1000)
    y_train = np.sin(x_train)
    # Neural network model
    model = Sequential([
        Dense(5, input_shape=(1,), activation='relu'),
        Dense(5, activation='relu'),
        Dense(5, activation='relu'),
        Dense(5, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(x_train, y_train, epochs=150, verbose=0)

    # Generate test data
    x_test = np.linspace(0, 2 * np.pi, 100)
    y_test = np.sin(x_test)
    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()
    model.summary()
    print(f"model 3 max: {np.max(abs(y_test - y_pred))}")
    error = np.sqrt(2 * np.pi * np.sum((y_test - y_pred) ** 2) / 100)  # Step 1: Calculate differences
    print(f"model 3 loss:{error}")

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(x_test, y_test)
    ax.plot(x_test, y_pred, linestyle='--')
    ax.set_title('Function and its Approximation')
    ax.legend()

    # Add an inset - zoom into the range [-2, 2] on x and [0, 10] on y
    ax_inset = inset_axes(ax, width="30%", height="30%", loc=3)  # loc=3 is lower left corner
    ax_inset.plot(x_test, y_test)
    ax_inset.plot(x_test, y_pred, linestyle='--')
    ax_inset.set_xlim(1.5, 2)  # Limits for x-axis
    ax_inset.set_ylim(0.75, 1.25)  # Limits for y-axis
    ax_inset.set_title('Zoom')

    # Optionally add marks that connect the area of the zoom to the larger plot
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    # Show the plot
    plt.show()
# Note: Uncomment the line below for execution and visualization in an actual Python environment.
# This plot visualizes the comparison between the true sin(x) function and its neural network approximation.

model1()
model2()
model3()
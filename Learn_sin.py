import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Generate dataset
x_train = np.linspace(0, 2*np.pi, 1000)
y_train = np.sin(x_train)

# Neural network model
model = Sequential([
    Dense(10, input_shape=(1,), activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(x_train, y_train, epochs=100, verbose=0)

# Generate test data
x_test = np.linspace(0, 2*np.pi, 100)
y_test = np.sin(x_test)
y_pred = model.predict(x_test)

# Plot results
plt.plot(x_test, y_test, label='True Function')
plt.plot(x_test, y_pred, label='NN Approximation', linestyle='--')
plt.legend()
plt.show()

# Note: Uncomment the line below for execution and visualization in an actual Python environment.
# This plot visualizes the comparison between the true sin(x) function and its neural network approximation.

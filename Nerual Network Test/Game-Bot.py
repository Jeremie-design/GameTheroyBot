import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


training_input = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
rand_numbers = np.random.random((3, 1))
scaled_up = 2 * rand_numbers
shifted = scaled_up - 1
synaptic_weights = shifted

print("Random starting synaptic weights")
print(synaptic_weights)

for iteration in range(10000):

    input_layer = training_input
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs
    adjustment = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustment)

print("Synaptic weights after training")
print(synaptic_weights)

print("Outputs after training")
round_outputs = np.round(outputs)
print(round_outputs)

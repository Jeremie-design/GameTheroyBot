import numpy as np


class MLPrisonerBot:

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def act(self, x1, x2):
        input_layer = np.array([[x1, x2, 1]])
        outputs = self.sigmoid(np.dot(input_layer, self.synaptic_weights))

        return int(np.round(outputs.item()))

    def train(self, training_inputs, training_targets, iterations=1):

        for _ in range(iterations):

            output = self.sigmoid(np.dot(training_inputs, self.synaptic_weights))

            error = training_targets - output
            adjustment = error * self.sigmoid_derivative(output)
            self.synaptic_weights += np.dot(training_inputs.T, adjustment)

    def learn(self, last_bot_move, last_opp_move, reward, lr=3.0):
        target = reward / 1.0

        inputs = np.array([[last_bot_move, last_opp_move, 1]])

        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        error = target - outputs
        adjustment = error * self.sigmoid_derivative(outputs)

        self.synaptic_weights += lr * np.dot(inputs.T, adjustment)

    def __repr__(self):
        return f"MLPrisonerBot(weights={self.synaptic_weights.flatten()})"

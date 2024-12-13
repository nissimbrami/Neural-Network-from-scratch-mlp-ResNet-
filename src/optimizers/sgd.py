# src/optimizers/sgd.py
import numpy as np


class SGDOptimizer:
    def __init__(self, learning_rate=0.005, momentum=0.9):
        """
        Initialize SGD optimizer with momentum
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self, parameters, gradients):

        if not self.velocities:
            for key in parameters:
                self.velocities[key] = np.zeros_like(parameters[key])

        for key in parameters:
            grad_key = 'd' + key  # למשל: 'W1' -> 'dW1'

            self.velocities[key] = (self.momentum * self.velocities[key] -
                                    self.learning_rate * gradients[grad_key])
            parameters[key] += self.velocities[key]

        return parameters

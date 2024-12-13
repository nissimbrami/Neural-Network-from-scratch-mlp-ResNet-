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
        """
        בצע צעד אופטימיזציה אחד

        Parameters:
        parameters: מילון של פרמטרים ('W1', 'b1', 'W2', 'b2')
        gradients: מילון של גרדיאנטים ('dW1', 'db1', 'dW2', 'db2')

        Returns:
        parameters: הפרמטרים המעודכנים
        """
        # אתחול velocities אם זה הצעד הראשון
        if not self.velocities:
            for key in parameters:
                self.velocities[key] = np.zeros_like(parameters[key])

        # עדכון כל פרמטר
        for key in parameters:
            # מתאים את שם הגרדיאנט לשם הפרמטר
            grad_key = 'd' + key  # למשל: 'W1' -> 'dW1'

            # עדכון ה-velocity וה-parameter
            self.velocities[key] = (self.momentum * self.velocities[key] -
                                    self.learning_rate * gradients[grad_key])
            parameters[key] += self.velocities[key]

        return parameters
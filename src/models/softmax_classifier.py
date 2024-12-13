import numpy as np


class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)

    def softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def forward(self, X):
        scores = np.dot(X, self.W) + self.b
        probs = self.softmax(scores)
        return probs

    def loss(self, X, y):
        num_train = X.shape[0]
        probs = self.forward(X)
        correct_logprobs = -np.log(probs[range(num_train), y])
        loss = np.sum(correct_logprobs) / num_train

        dscores = probs.copy()
        dscores[range(num_train), y] -= 1
        dscores /= num_train

        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0)

        return loss, {'W': dW, 'b': db}

    def gradient_check(self, X, y, epsilon=1e-7):
        loss, gradients = self.loss(X, y)
        analytic_grad_W = gradients['W']

        numerical_grad_W = np.zeros_like(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j] += epsilon
                loss_plus, _ = self.loss(X, y)

                self.W[i, j] -= 2 * epsilon
                loss_minus, _ = self.loss(X, y)

                self.W[i, j] += epsilon

                numerical_grad_W[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        diff = np.linalg.norm(analytic_grad_W - numerical_grad_W) / np.linalg.norm(analytic_grad_W + numerical_grad_W)
        return diff

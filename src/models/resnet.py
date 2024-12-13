import numpy as np


class ResNet:
    def __init__(self, layer_dims, activation='relu', regularization=None, reg_lambda=0.0, use_dropout=True,
                 dropout_rate=0.1):

        self.L = len(layer_dims) - 1
        self.layer_dims = layer_dims
        self.activation = activation
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.cache = {}
        self.parameters = {}

        dim = layer_dims[0]
        for l in range(1, self.L):
            scale = np.sqrt(1. / dim)
            self.parameters[f'W1_{l}'] = np.random.randn(dim, dim) * scale
            self.parameters[f'b1_{l}'] = np.zeros((dim, 1))
            self.parameters[f'W2_{l}'] = np.random.randn(dim, dim) * scale * 0.1
            self.parameters[f'b2_{l}'] = np.zeros((dim, 1))

            self.parameters[f'gamma_{l}'] = np.ones((dim, 1))
            self.parameters[f'beta_{l}'] = np.zeros((dim, 1))

            self.cache[f'bn_1_{l}_running_mean'] = np.zeros((dim, 1))
            self.cache[f'bn_1_{l}_running_var'] = np.ones((dim, 1))
            self.cache[f'bn_2_{l}_running_mean'] = np.zeros((dim, 1))
            self.cache[f'bn_2_{l}_running_var'] = np.ones((dim, 1))

        self.parameters[f'W_{self.L}'] = np.random.randn(layer_dims[-1], dim) * np.sqrt(2. / dim)
        self.parameters[f'b_{self.L}'] = np.zeros((layer_dims[-1], 1))

    def batch_norm_forward(self, x, gamma, beta, layer_idx, mode='train'):

        if mode == 'train':
            mu = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True) + 1e-8
            x_norm = (x - mu) / np.sqrt(var)
            out = gamma * x_norm + beta

            momentum = 0.9
            running_mean_key = f'bn_{layer_idx}_running_mean'
            running_var_key = f'bn_{layer_idx}_running_var'

            self.cache[running_mean_key] = momentum * self.cache[running_mean_key] + (1 - momentum) * mu
            self.cache[running_var_key] = momentum * self.cache[running_var_key] + (1 - momentum) * var

            self.cache[f'bn_{layer_idx}'] = {
                'x_norm': x_norm,
                'gamma': gamma,
                'var': var,
                'mu': mu,
                'x': x
            }
        else:  
            x_norm = (x - self.cache[f'bn_{layer_idx}_running_mean']) / \
                     np.sqrt(self.cache[f'bn_{layer_idx}_running_var'] + 1e-8)
            out = gamma * x_norm + beta

        return out

    def batch_norm_backward(self, dout, cache):
        """
        Batch Normalization - Backward Pass
        """
        x_norm, gamma, var, mu, x = cache['x_norm'], cache['gamma'], cache['var'], cache['mu'], cache['x']
        m = x.shape[1]

        dgamma = np.sum(dout * x_norm, axis=1, keepdims=True)
        dbeta = np.sum(dout, axis=1, keepdims=True)

        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + 1e-8) ** (-1.5), axis=1, keepdims=True)
        dmu = np.sum(dx_norm * -1 / np.sqrt(var + 1e-8), axis=1, keepdims=True) + \
              dvar * np.mean(-2 * (x - mu), axis=1, keepdims=True)
        dx = dx_norm / np.sqrt(var + 1e-8) + dvar * 2 * (x - mu) / m + dmu / m

        return dx, dgamma, dbeta

    def activation_forward(self, Z):

        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'leaky_relu':
            return np.where(Z > 0, Z, 0.01 * Z)
        else:  # tanh
            return np.tanh(Z)

    def activation_backward(self, dA, Z):

        if self.activation == 'relu':
            return dA * (Z > 0)
        elif self.activation == 'leaky_relu':
            return np.where(Z > 0, dA, 0.01 * dA)
        else:  # tanh
            return dA * (1 - np.power(np.tanh(Z), 2))

    def dropout_forward(self, X, mask=None):

        if not self.use_dropout or mask is not None:
            return X, mask

        mask = (np.random.rand(*X.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return X * mask, mask

    def forward_prop(self, X, mode='train'):
        """
        Forward Propagation
        """
        A = X.T
        self.cache['A0'] = A

        for l in range(1, self.L):
            identity = A  # שמירת הקלט המקורי לחיבור השארית

            Z1 = np.dot(self.parameters[f'W1_{l}'], A) + self.parameters[f'b1_{l}']
            # Batch Norm אחרי הכפלה במטריצה
            Z1 = self.batch_norm_forward(Z1,
                                         self.parameters[f'gamma_{l}'],
                                         self.parameters[f'beta_{l}'],
                                         f'1_{l}',
                                         mode)
            A1 = self.activation_forward(Z1)

            # Dropout
            if mode == 'train' and self.use_dropout:
                A1, mask1 = self.dropout_forward(A1)
                self.cache[f'dropout1_{l}'] = mask1

            # חלק שני של הבלוק
            Z2 = np.dot(self.parameters[f'W2_{l}'], A1) + self.parameters[f'b2_{l}']

            A = self.activation_forward(identity + Z2)

            if mode == 'train' and self.use_dropout:
                A, mask2 = self.dropout_forward(A)
                self.cache[f'dropout2_{l}'] = mask2

            # שמירה במטמון
            self.cache[f'Z1_{l}'] = Z1
            self.cache[f'A1_{l}'] = A1
            self.cache[f'Z2_{l}'] = Z2
            self.cache[f'A_{l}'] = A
            self.cache[f'identity_{l}'] = identity

        ZL = np.dot(self.parameters[f'W_{self.L}'], A) + self.parameters[f'b_{self.L}']
        self.cache[f'Z_{self.L}'] = ZL

        return ZL.T

    def compute_cost(self, AL, Y):
    
        m = Y.shape[0]

        scores = AL - np.max(AL, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_log_probs = -np.log(probs[range(m), Y] + 1e-15)
        data_loss = np.sum(correct_log_probs) / m

        reg_loss = 0
        if self.regularization == 'l2':
            for l in range(1, self.L):
                reg_loss += (np.sum(np.square(self.parameters[f'W1_{l}'])) +
                             np.sum(np.square(self.parameters[f'W2_{l}'])))
            reg_loss += np.sum(np.square(self.parameters[f'W_{self.L}']))
            reg_loss *= self.reg_lambda / (2 * m)

        total_loss = data_loss + reg_loss

        dscores = probs.copy()
        dscores[range(m), Y] -= 1
        dscores /= m

        return total_loss, dscores.T

    def backward_prop(self, AL, Y):
     
        gradients = {}
        m = Y.shape[0]

        loss, dZ = self.compute_cost(AL, Y)

        l = self.L
        A_prev = self.cache[f'A_{l - 1}']
        gradients[f'dW_{l}'] = np.dot(dZ, A_prev.T)
        gradients[f'db_{l}'] = np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.parameters[f'W_{l}'].T, dZ)

        for l in reversed(range(1, self.L)):
            if self.use_dropout:
                dA *= self.cache[f'dropout2_{l}']

            dZ_block = self.activation_backward(dA, self.cache[f'Z2_{l}'] + self.cache[f'identity_{l}'])

            gradients[f'dW2_{l}'] = np.dot(dZ_block, self.cache[f'A1_{l}'].T)
            gradients[f'db2_{l}'] = np.sum(dZ_block, axis=1, keepdims=True)

            dA1 = np.dot(self.parameters[f'W2_{l}'].T, dZ_block)

            if self.use_dropout:
                dA1 *= self.cache[f'dropout1_{l}']

            dZ1 = self.activation_backward(dA1, self.cache[f'Z1_{l}'])

            # Batch Norm backward
            dZ1, dgamma, dbeta = self.batch_norm_backward(dZ1, self.cache[f'bn_1_{l}'])
            gradients[f'dgamma_{l}'] = dgamma
            gradients[f'dbeta_{l}'] = dbeta

            A_prev = self.cache[f'A_{l - 1}'] if l > 1 else self.cache['A0']
            gradients[f'dW1_{l}'] = np.dot(dZ1, A_prev.T)
            gradients[f'db1_{l}'] = np.sum(dZ1, axis=1, keepdims=True)

            dA = np.dot(self.parameters[f'W1_{l}'].T, dZ1) + dZ_block

            if self.regularization == 'l2':
                gradients[f'dW1_{l}'] += (self.reg_lambda / m) * self.parameters[f'W1_{l}']
                gradients[f'dW2_{l}'] += (self.reg_lambda / m) * self.parameters[f'W2_{l}']

        return gradients, loss

    def train_step(self, X, Y):
     
        AL = self.forward_prop(X, mode='train')
        gradients, cost = self.backward_prop(AL, Y)
        return gradients, cost

    def predict(self, X):

        scores = self.forward_prop(X, mode='test')
        return np.argmax(scores, axis=1)

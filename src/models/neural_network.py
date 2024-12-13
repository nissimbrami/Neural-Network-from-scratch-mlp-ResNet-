import numpy as np


class NeuralNetwork:
    def __init__(self, layer_dims, activation='relu', regularization='l2', reg_lambda=0.01):
        self.L = len(layer_dims) - 1
        self.layer_dims = layer_dims
        self.activation = activation
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.parameters = {}
        self.cache = {}

        for l in range(1, self.L + 1):
            if activation == 'tanh':
                factor = np.sqrt(1. / layer_dims[l - 1])
            else:  # relu
                factor = np.sqrt(2. / layer_dims[l - 1])
            self.parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * factor
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    def activation_forward(self, Z):
        if self.activation == 'tanh':
            return np.tanh(Z)
        else:  # relu
            return np.maximum(0, Z)

    def activation_backward(self, dA, Z):
        if self.activation == 'tanh':
            return dA * (1 - np.power(np.tanh(Z), 2))
        else:  # relu
            return dA * (Z > 0)

    def linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, layer_num):
        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if layer_num == self.L:
            A = Z
        else:
            A = self.activation_forward(Z)

        cache = (linear_cache, Z)
        return A, cache

    def forward_prop(self, X):
        A = X.T
        self.cache['A0'] = A

        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            A, cache = self.linear_activation_forward(A_prev, W, b, l)
            self.cache[f'layer{l}'] = cache

        return A.T


    def compute_cost(self, AL, Y):
        m = Y.shape[0]

        scores = AL
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_log_probs = -np.log(probs[range(m), Y] + 1e-15)
        data_loss = np.sum(correct_log_probs) / m

        reg_loss = 0
        if self.regularization == 'l2':
            for l in range(1, self.L + 1):
                reg_loss += np.sum(np.square(self.parameters[f'W{l}']))
            reg_loss = (self.reg_lambda / (2 * m)) * reg_loss

        total_loss = data_loss + reg_loss

        # חישוב הגרדיאנט
        dscores = probs.copy()
        dscores[range(m), Y] -= 1
        dscores /= m

        return total_loss, dscores.T

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, layer_num):
        linear_cache, Z = cache

        if layer_num == self.L:
            dZ = dA
        else:
            dZ = self.activation_backward(dA, Z)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def backward_prop(self, AL, Y):
        gradients = {}
        m = Y.shape[0]

        cost, dAL = self.compute_cost(AL, Y)

        current_cache = self.cache[f'layer{self.L}']
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, self.L)

        if self.regularization == 'l2':
            dW_temp += (self.reg_lambda / m) * self.parameters[f'W{self.L}']

        gradients[f'dW{self.L}'] = dW_temp
        gradients[f'db{self.L}'] = db_temp

        for l in reversed(range(1, self.L)):
            current_cache = self.cache[f'layer{l}']
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, l)

            if self.regularization == 'l2':
                dW_temp += (self.reg_lambda / m) * self.parameters[f'W{l}']

            gradients[f'dW{l}'] = dW_temp
            gradients[f'db{l}'] = db_temp

        return gradients, cost

    def train_step(self, X, Y):
     
        AL = self.forward_prop(X)
        gradients, cost = self.backward_prop(AL, Y)
        return gradients, cost


    def gradient_check(self, X, Y, epsilon=1e-7):
   
        tolerance = np.sqrt(epsilon)

        print("\n" + "=" * 50)
        print(f"בדיקת גרדיאנט (epsilon={epsilon:.1e}, tolerance={tolerance:.1e})")
        print("=" * 50)

        params_values = {}
        for key in self.parameters:
            params_values[key] = self.parameters[key].copy()

        gradients, cost = self.train_step(X, Y)
        print(f"\nעלות התחלתית: {cost:.8f}")

        total_params = 0
        total_checks = 0
        failed_checks = 0

        for param_name in self.parameters:
            grad_name = f'd{param_name}'
            param = self.parameters[param_name]
            grad = gradients[grad_name]

            print(f"\n{'-' * 20} {param_name} {'-' * 20}")
            print(f"צורה: {param.shape}")
            print(f"טווח פרמטרים: [{param.min():.8f}, {param.max():.8f}]")
            print(f"טווח גרדיאנטים: [{grad.min():.8f}, {grad.max():.8f}]")

            num_checks = min(100, param.size) if param.size < 1000 else min(50, param.size)
            idx = np.random.choice(param.size, num_checks, replace=False)

            failures = []  # נשמור את השגיאות לתצוגה מרוכזת

            for i in idx:
                ix = np.unravel_index(i, param.shape)
                old_value = param[ix]
                total_params += 1

                param[ix] = old_value + epsilon
                _, cost_plus = self.train_step(X, Y)

                param[ix] = old_value - epsilon
                _, cost_minus = self.train_step(X, Y)

                param[ix] = old_value

                grad_numerical = (cost_plus - cost_minus) / (2 * epsilon)
                grad_analytic = grad[ix]

                if abs(grad_numerical) > 1e-8 or abs(grad_analytic) > 1e-8:
                    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
                    total_checks += 1

                    if rel_error > tolerance:
                        failed_checks += 1
                        failures.append({
                            'location': ix,
                            'numerical': grad_numerical,
                            'analytic': grad_analytic,
                            'ratio': grad_analytic / grad_numerical if grad_numerical != 0 else 'N/A',
                            'error': rel_error
                        })

            if failures:
                print(f"\nנמצאו {len(failures)} שגיאות:")
                print("-" * 60)
                print(f"{'מיקום':^15} {'נומרי':^15} {'אנליטי':^15} {'שגיאה':^12}")
                print("-" * 60)
                for fail in failures[:5]:  # מציג רק 5 הראשונות
                    print(
                        f"{str(fail['location']):^15} {fail['numerical']:^15.8f} {fail['analytic']:^15.8f} {fail['error']:.2e}")
                if len(failures) > 5:
                    print(f"... ועוד {len(failures) - 5} שגיאות נוספות")

            print(f"\nסיכום {param_name}:")
            print(f"מספר בדיקות: {total_checks}")
            print(f"מספר שגיאות: {len(failures)}")

        self.parameters = params_values

        print("\n" + "=" * 50)
        print("סיכום בדיקת גרדיאנט:")
        print("=" * 50)
        success_rate = (total_checks - failed_checks) / total_checks * 100 if total_checks > 0 else 0
        print(f"סה״כ נבדקו: {total_checks} פרמטרים")
        print(f"אחוז הצלחה: {success_rate:.2f}%")
        print("=" * 50)

    def comprehensive_gradient_check(self, X, Y, epsilon=1e-7):
  
        print("\nמתחיל בדיקת גרדיאנט מקיפה...")

        for l in range(1, self.L + 1):
            print(f"\nבודק שכבה {l}:")
            A_prev = self.cache[f'A{l - 1}']
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            old_activation = self.activation
            self.activation = 'tanh'

            Z, cache = self.linear_forward(A_prev, W, b)
            A = self.activation_forward(Z)

            dA = np.random.randn(*A.shape)

            dZ = self.activation_backward(dA, Z)
            dA_prev, dW, db = self.linear_backward(dZ, cache)

            from src.utils.gradient_tests import plot_slope_test

            def f_W(W_test):
                _, cache = self.linear_forward(A_prev, W_test, b)
                A_test = self.activation_forward(cache[2])
                return np.sum(A_test * dA)

            plot_slope_test(f_W, W, dW, f'Layer {l} - Weights')

            self.activation = old_activation

        print("\nבודק את הרשת השלמה:")

        gradients, cost = self.train_step(X, Y)

        for param_name, param in self.parameters.items():
            grad = gradients['d' + param_name]

            def f_full(param_test):
                old_param = self.parameters[param_name].copy()
                self.parameters[param_name] = param_test
                AL = self.forward_prop(X)
                loss, _ = self.compute_cost(AL, Y)
                self.parameters[param_name] = old_param
                return loss

            plot_slope_test(f_full, param, grad, f'Full Network - {param_name}')

    def compute_cost(self, AL, Y):

        m = Y.shape[0]

        scores = AL - np.max(AL, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_log_probs = -np.log(probs[range(m), Y] + 1e-15)
        data_loss = np.sum(correct_log_probs) / m

        reg_loss = 0
        if self.regularization == 'l2':
            for l in range(1, self.L + 1):
                reg_loss += np.sum(np.square(self.parameters[f'W{l}']))
            reg_loss = (self.reg_lambda / (2 * m)) * reg_loss

        total_loss = data_loss + reg_loss

        dscores = probs.copy()
        dscores[range(m), Y] -= 1
        # dscores /= m  <- מסירים את השורה הזו

        return total_loss, dscores.T

    def compute_accuracy(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)

    def predict(self, X):

        scores = self.forward_prop(X)

        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return np.argmax(probs, axis=1)

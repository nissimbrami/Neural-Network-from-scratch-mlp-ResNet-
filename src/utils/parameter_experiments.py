import numpy as np
from ..models.neural_network import NeuralNetwork
from ..models.resnet import ResNet


class ParameterExperiments:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_params = 100 * num_classes

    def count_parameters(self, layer_dims, is_resnet=False):
  
        total = 0
        for i in range(len(layer_dims) - 1):
            if is_resnet and i < len(layer_dims) - 2:
                # ResNet block
                total += (layer_dims[i] * layer_dims[i + 1] + layer_dims[i + 1]) * 2
            else:
                # Regular layer
                total += layer_dims[i] * layer_dims[i + 1] + layer_dims[i + 1]
        return total

    def design_networks(self):
   
        networks = []

        dims = [self.input_dim]
        curr_dim = self.input_dim
        total_params = 0

        while total_params < self.max_params * 0.9:
            next_dim = max(curr_dim // 2, 2)
            if total_params + (curr_dim * next_dim + next_dim) > self.max_params:
                break
            dims.append(next_dim)
            total_params += curr_dim * next_dim + next_dim
            curr_dim = next_dim

        dims.append(self.num_classes)
        networks.append(('Narrow Deep', dims, False))

        hidden_dim = int(np.sqrt(self.max_params / 2))
        dims = [self.input_dim, hidden_dim, self.num_classes]
        networks.append(('Wide Shallow', dims, False))

        resnet_dims = [self.input_dim]
        hidden_dim = max(self.input_dim, self.num_classes)
        while self.count_parameters(resnet_dims + [hidden_dim, self.num_classes], True) < self.max_params:
            resnet_dims.append(hidden_dim)
        resnet_dims.append(self.num_classes)
        networks.append(('Minimal ResNet', resnet_dims, True))

        return networks

    def run_experiments(self, X_train, y_train, X_val, y_val):
      
        from ..optimizers.sgd import SGDOptimizer
        from .trainer import NetworkTrainer

        results = {}
        networks = self.design_networks()

        for name, dims, is_resnet in networks:
            print(f"\nRunning experiment: {name}")
            print(f"Architecture: {dims}")
            print(f"Total parameters: {self.count_parameters(dims, is_resnet)}")

            if is_resnet:
                model = ResNet(dims)
            else:
                model = NeuralNetwork(dims)

            optimizer = SGDOptimizer(learning_rate=0.001, momentum=0.9)
            trainer = NetworkTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=32,
                n_epochs=100
            )

            # אימון
            history = trainer.train(X_train, y_train, X_val, y_val)
            results[name] = history

        return results

# src/utils/experiment.py
import numpy as np
import matplotlib.pyplot as plt
from ..models.neural_network import NeuralNetwork
from ..models.resnet import ResNet
from ..optimizers.sgd import SGDOptimizer
from .trainer import NetworkTrainer


class ExperimentRunner:
    def __init__(self, X_train, y_train, X_val, y_val):
        """
        מחלקה להרצת ניסויים עם פרמטרים שונים

        Parameters:
        X_train, y_train: נתוני אימון
        X_val, y_val: נתוני ולידציה
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.results = {}

    def run_experiment(self, model_type, layer_dims, name, **kwargs):
        """
        הרצת ניסוי בודד

        Parameters:
        model_type: "standard" או "resnet"
        layer_dims: רשימת גדלי השכבות
        name: שם הניסוי
        **kwargs: פרמטרים נוספים (learning_rate, batch_size וכו')
        """
        print(f"\nRunning experiment: {name}")
        print("Architecture:", layer_dims)
        print("Parameters:", kwargs)

        # יצירת המודל
        if model_type == "standard":
            model = NeuralNetwork(layer_dims, activation=kwargs.get('activation', 'tanh'))
        else:  # resnet
            model = ResNet(layer_dims, activation=kwargs.get('activation', 'relu'))

        # הגדרת אופטימייזר
        optimizer = SGDOptimizer(
            learning_rate=kwargs.get('learning_rate', 0.01),
            momentum=kwargs.get('momentum', 0.9)
        )

        # הגדרת מאמן
        trainer = NetworkTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=kwargs.get('batch_size', 32),
            n_epochs=kwargs.get('n_epochs', 50),
            verbose=True
        )

        # אימון
        history = trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )

        # שמירת תוצאות
        self.results[name] = {
            'model': model,
            'history': history,
            'params': kwargs,
            'layer_dims': layer_dims
        }

        return trainer

    def compare_results(self):
        """
        השוואת תוצאות הניסויים
        """
        plt.figure(figsize=(15, 5))

        # השוואת דיוק
        plt.subplot(1, 2, 1)
        for name, result in self.results.items():
            plt.plot(result['history']['val_acc'], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy Comparison')
        plt.legend()
        plt.grid(True)

        # השוואת loss
        plt.subplot(1, 2, 2)
        for name, result in self.results.items():
            plt.plot(result['history']['train_loss'], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Loss Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # הדפסת סיכום
        print("\nFinal Results:")
        for name, result in self.results.items():
            history = result['history']
            print(f"\n{name}:")
            print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
            print(f"Architecture: {result['layer_dims']}")
            print(f"Parameters: {result['params']}")

    def get_best_model(self, metric='val_acc'):
        """
        החזרת המודל הטוב ביותר לפי מדד מסוים
        """
        best_value = -float('inf')
        best_name = None

        for name, result in self.results.items():
            history = result['history']
            final_value = history[metric][-1]

            if final_value > best_value:
                best_value = final_value
                best_name = name

        if best_name:
            return self.results[best_name]['model'], best_value
        return None, None
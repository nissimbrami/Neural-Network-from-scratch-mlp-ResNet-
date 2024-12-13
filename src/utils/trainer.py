# src/utils/trainer.py
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from tqdm import tqdm
import time

from src.models.resnet import ResNet
from src.models.neural_network import NeuralNetwork
from src.optimizers.sgd import SGDOptimizer


class NetworkTrainer:
    def __init__(self, model, optimizer, batch_size=32, n_epochs=100, verbose=True):
        """
        מאמן לרשתות נוירונים

        Parameters:
        model: הרשת (NeuralNetwork או ResNet)
        optimizer: האופטימייזר (SGD)
        batch_size: גודל ה-batch
        n_epochs: מספר אפוקות
        verbose: האם להציג התקדמות
        """
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def create_mini_batches(self, X, y):
        """
        יצירת mini-batches מהנתונים
        """
        m = X.shape[0]
        mini_batches = []

        # ערבוב הנתונים
        permutation = np.random.permutation(m)
        shuffled_X = X[permutation]
        shuffled_y = y[permutation]

        # חלוקה ל-batches
        n_complete_batches = m // self.batch_size
        for k in range(n_complete_batches):
            batch_X = shuffled_X[k * self.batch_size:(k + 1) * self.batch_size]
            batch_y = shuffled_y[k * self.batch_size:(k + 1) * self.batch_size]
            mini_batches.append((batch_X, batch_y))

        # טיפול בשארית אם יש
        if m % self.batch_size != 0:
            batch_X = shuffled_X[n_complete_batches * self.batch_size:]
            batch_y = shuffled_y[n_complete_batches * self.batch_size:]
            mini_batches.append((batch_X, batch_y))

        return mini_batches

    def compute_accuracy(self, X, y):
        """
        חישוב דיוק המודל

        Args:
            X: נתוני קלט
            y: תוויות אמת
        """
        predictions = self.model.predict(X)

        # נרמול הצורות
        if len(y.shape) > 1:
            y = y.ravel()
        if len(predictions.shape) > 1:
            predictions = predictions.ravel()

        # וידוא שהצורות תואמות
        if predictions.shape[0] != y.shape[0]:
            print(f"Warning: Predictions shape: {predictions.shape}, y shape: {y.shape}")
            min_len = min(predictions.shape[0], y.shape[0])
            predictions = predictions[:min_len]
            y = y[:min_len]

        return np.mean(predictions == y)

    def predict(self, X):
        """
        חיזוי על נתונים חדשים
        """
        # Forward pass
        output = self.model.forward_prop(X)

        # המרה להסתברויות
        exp_scores = np.exp(output)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # בחירת המחלקה עם ההסתברות הגבוהה ביותר
        return np.argmax(probs, axis=1)

    # src/utils/trainer.py
    # [previous imports remain the same]

    from colorama import init, Fore, Style
    init()  # מאתחל את colorama

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        אימון המודל עם הדפסות צבעוניות ומשופרות
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        print("\n" + Fore.CYAN + "=" * 60 + Style.RESET_ALL)
        print(Fore.CYAN + f"{'תחילת אימון':^60}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 60 + Style.RESET_ALL)

        for epoch in range(1, self.n_epochs + 1):
            # כותרת אפוק
            print(f"\n{Fore.GREEN}Epoch {epoch}/{self.n_epochs}{Style.RESET_ALL}")
            print(Fore.BLUE + "-" * 60 + Style.RESET_ALL)

            epoch_loss = 0
            batch_count = 0

            mini_batches = self.create_mini_batches(X_train, y_train)
            n_batches = len(mini_batches)

            for batch_idx, (batch_X, batch_y) in enumerate(mini_batches, 1):
                gradients, loss = self.model.train_step(batch_X, batch_y)
                epoch_loss += loss

                self.model.parameters = self.optimizer.step(
                    self.model.parameters,
                    gradients
                )

                if self.verbose and batch_idx % max(1, n_batches // 20) == 0:
                    progress = batch_idx / n_batches
                    bar_length = 30
                    filled_length = int(bar_length * progress)

                    # פרוגרס בר צבעוני
                    bar = (Fore.GREEN + '█' * filled_length +
                           Fore.WHITE + '░' * (bar_length - filled_length))
                    print(f"\r{Fore.BLUE}[{bar}{Fore.BLUE}]{Style.RESET_ALL} "
                          f"{Fore.YELLOW}{progress * 100:>3.0f}%{Style.RESET_ALL} | "
                          f"Loss: {Fore.RED}{loss:.4f}{Style.RESET_ALL}", end='')

            avg_loss = epoch_loss / n_batches
            train_acc = self.compute_accuracy(X_train, y_train)

            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)

            # מדדי ולידציה
            val_metrics = ""
            if X_val is not None and y_val is not None:
                val_acc = self.compute_accuracy(X_val, y_val)
                _, val_loss = self.model.train_step(X_val, y_val)
                history['val_acc'].append(val_acc)
                history['val_loss'].append(val_loss)
                val_metrics = (f" | {Fore.MAGENTA}Val Loss: {val_loss:.4f} | "
                               f"Val Acc: {val_acc:.4f}{Style.RESET_ALL}")

            # סיכום אפוק צבעוני
            print(f"\r{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f}{val_metrics}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'-' * 60}{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(Fore.CYAN + f"{'סיום אימון':^60}" + Style.RESET_ALL)
        print(Fore.CYAN + "=" * 60 + Style.RESET_ALL)

        return history

    def plot_training_history(self):
        """
        יצירת גרפים של מהלך האימון
        """
        plt.figure(figsize=(15, 5))

        # גרף Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # גרף Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        if 'val_acc' in self.history and len(self.history['val_acc']) > 0:
            plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# src/utils/experiment.py
class ExperimentRunner:
    """
    מחלקה להרצת ניסויים עם פרמטרים שונים
    """

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.results = {}

    def run_experiment(self, model_type, layer_dims, name, **kwargs):
        """
        הרצת ניסוי בודד
        """
        print(f"\nRunning experiment: {name}")
        print("Architecture:", layer_dims)
        print("Parameters:", kwargs)

        # יצירת המודל
        if model_type == "standard":
            model = NeuralNetwork(layer_dims, **kwargs)
        else:  # resnet
            model = ResNet(layer_dims, **kwargs)

        # הגדרת אופטימייזר ומאמן
        optimizer = SGDOptimizer(learning_rate=kwargs.get('learning_rate', 0.01),
                                 momentum=kwargs.get('momentum', 0.9))

        trainer = NetworkTrainer(
            model, optimizer,
            batch_size=kwargs.get('batch_size', 32),
            n_epochs=kwargs.get('n_epochs', 100)
        )

        # אימון
        history = trainer.train(self.X_train, self.y_train,
                                self.X_val, self.y_val)

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
            plt.plot(result['history']['val_loss'], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Loss Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # הדפסת סיכום
        print("\nFinal Results:")
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"Final validation accuracy: {result['history']['val_acc'][-1]:.4f}")
            print(f"Architecture: {result['layer_dims']}")
            print(f"Parameters: {result['params']}")
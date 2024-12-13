import os
import numpy as np
import matplotlib.pyplot as plt
from src.models.neural_network import NeuralNetwork
from src.models.resnet import ResNet
from src.optimizers.sgd import SGDOptimizer
from src.utils.data_loader import DataLoader
from src.utils.visualization import plot_training_results


def get_data_path():
    """
    מציאת נתיב נכון לקבצי הנתונים
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data')


def test_overfitting_capability():
    """
    בדיקת יכולת הרשת להתכנס לדאטא קטן
    """
    print("\n=== בדיקת יכולת התכנסות ===")

    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    standard_net = NeuralNetwork([2, 32, 16, 2], activation='relu')
    optimizer = SGDOptimizer(learning_rate=0.001, momentum=0.9)

    losses = []
    accuracies = []

    for _ in range(1000):
        gradients, loss = standard_net.train_step(X, y)
        standard_net.parameters = optimizer.step(standard_net.parameters, gradients)
        predictions = standard_net.predict(X)
        accuracy = np.mean(predictions == y)
        losses.append(loss)
        accuracies.append(accuracy)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(losses)
    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(122)
    plt.plot(accuracies)
    plt.title('Accuracy vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()

    final_accuracy = accuracies[-1]
    print(f"דיוק סופי על סט האימון: {final_accuracy:.4f}")
    assert final_accuracy > 0.95, "הרשת צריכה להגיע לדיוק גבוה על סט קטן"


def test_regularization():
    """
    בדיקת השפעת רגולריזציה
    """
    print("\n=== בדיקת רגולריזציה ===")

    X = np.random.randn(100, 2)
    y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) > 0).astype(int)

    reg_lambdas = [0, 0.1, 1.0]
    results = {}

    for reg_lambda in reg_lambdas:
        net = NeuralNetwork(
            [2, 32, 32, 2],
            activation='relu',
            regularization='l2',
            reg_lambda=reg_lambda
        )
        optimizer = SGDOptimizer(learning_rate=0.01)

        train_losses = []
        for _ in range(200):
            gradients, loss = net.train_step(X, y)
            net.parameters = optimizer.step(net.parameters, gradients)
            train_losses.append(loss)

        results[reg_lambda] = train_losses
        print(f"רגולריזציה lambda={reg_lambda}, loss סופי: {train_losses[-1]:.4f}")

    plt.figure(figsize=(10, 5))
    for reg_lambda, losses in results.items():
        plt.plot(losses, label=f'lambda={reg_lambda}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('השפעת רגולריזציה על ה-Loss')
    plt.legend()
    plt.show()


def compare_architectures():
    """
    השוואה בין ארכיטקטורות שונות
    """
    print("\n=== השוואת ארכיטקטורות ===")

    data_path = os.path.join(get_data_path(), 'SwissRollData.mat')
    data_loader = DataLoader(data_path)
    (X_train, y_train), (X_val, y_val) = data_loader.get_data()

    # נרמול נכון של הנתונים
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    architectures = {
        'Shallow': NeuralNetwork(
            [2, 32, 2],
            activation='relu',
            reg_lambda=0.01
        ),
        'Deep': NeuralNetwork(
            [2, 64, 32, 16, 8, 2],
            activation='relu',
            reg_lambda=0.01
        ),
        'Wide': NeuralNetwork(
            [2, 128, 64, 2],
            activation='relu',
            reg_lambda=0.01
        ),
        'ResNet': ResNet(
            layer_dims=[2, 128, 2],
            activation='relu',
            reg_lambda=0.001
        )
    }

    results = {}

    for name, model in architectures.items():
        print(f"\nבודק ארכיטקטורה: {name}")
        optimizer = SGDOptimizer(
            learning_rate=0.001 if name != 'ResNet' else 0.0005,
            momentum=0.9
        )

        val_accuracies = []
        train_accuracies = []
        losses = []

        n_epochs = 100
        batch_size = 128

        try:
            for epoch in range(n_epochs):
                epoch_losses = []
                indices = np.random.permutation(len(X_train_norm))

                for i in range(0, len(X_train_norm), batch_size):
                    batch_idx = indices[i:i + batch_size]
                    X_batch = X_train_norm[batch_idx]
                    y_batch = y_train[batch_idx]

                    gradients, loss = model.train_step(X_batch, y_batch)
                    model.parameters = optimizer.step(model.parameters, gradients)
                    epoch_losses.append(loss)

                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)

                if epoch % 10 == 0:
                    train_pred = model.predict(X_train_norm)
                    train_acc = np.mean(train_pred == y_train)
                    train_accuracies.append(train_acc)

                    val_pred = model.predict(X_val_norm)
                    val_acc = np.mean(val_pred == y_val)
                    val_accuracies.append(val_acc)

                    print(f"Epoch {epoch}: Train acc={train_acc:.4f}, Val acc={val_acc:.4f}, Loss={avg_loss:.4f}")

            results[name] = {
                'train_acc': train_accuracies,
                'val_acc': val_accuracies,
                'loss': losses
            }

            print(f"דיוק סופי באימון: {train_accuracies[-1]:.4f}")
            print(f"דיוק סופי בולידציה: {val_accuracies[-1]:.4f}")

        except Exception as e:
            print(f"שגיאה באימון {name}: {str(e)}")
            continue

    if results:
        plot_training_results(results)


def test_batch_effects():
    """
    בדיקת השפעת גודל ה-batch
    """
    print("\n=== בדיקת השפעת גודל ה-batch ===")

    X = np.random.randn(1000, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    batch_sizes = [16, 64, 256]
    results = {}

    for batch_size in batch_sizes:
        model = NeuralNetwork([2, 32, 2], activation='relu')
        optimizer = SGDOptimizer(learning_rate=0.01)

        losses = []
        for _ in range(100):
            indices = np.random.choice(X.shape[0], batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            gradients, loss = model.train_step(X_batch, y_batch)
            model.parameters = optimizer.step(model.parameters, gradients)
            losses.append(loss)

        results[batch_size] = losses
        print(f"Batch size {batch_size}, loss סופי: {losses[-1]:.4f}")

    plt.figure(figsize=(10, 5))
    for batch_size, losses in results.items():
        plt.plot(losses, label=f'batch_size={batch_size}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('השפעת גודל ה-batch')
    plt.legend()
    plt.show()


def main():
    """
    הרצת כל הבדיקות
    """
    try:
        test_overfitting_capability()
        test_regularization()
        compare_architectures()
        test_batch_effects()
        print("\nכל הבדיקות הושלמו בהצלחה!")
    except Exception as e:
        print(f"שגיאה במהלך הבדיקות: {str(e)}")
        raise


if __name__ == "__main__":
    main()
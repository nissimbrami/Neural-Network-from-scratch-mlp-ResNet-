# src/test_simple.py
import numpy as np
import matplotlib.pyplot as plt
from models.neural_network import NeuralNetwork
from optimizers.sgd import SGDOptimizer
from src.utils.visualization import plot_training_results
from utils.trainer import NetworkTrainer
from src.utils.gradient_tests import least_squares_example
from src.utils.parameter_experiments import ParameterExperiments
from utils.visualization import plot_training_results, plot_data_and_decision_boundary


def create_simple_dataset():
    """
    יצירת נתוני בדיקה פשוטים: שני מעגלים קונצנטריים
    """
    np.random.seed(42)
    n_samples = 100

    # יצירת מעגלים
    t = np.linspace(0, 2 * np.pi, n_samples // 2)
    r1 = np.random.normal(2, 0.2, n_samples // 2)
    r2 = np.random.normal(4, 0.2, n_samples // 2)

    # יצירת נקודות
    inner_x = r1 * np.cos(t)
    inner_y = r1 * np.sin(t)
    outer_x = r2 * np.cos(t)
    outer_y = r2 * np.sin(t)

    # איחוד הנתונים
    X = np.vstack([
        np.column_stack([inner_x, inner_y]),
        np.column_stack([outer_x, outer_y])
    ])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # חלוקה לאימון ובדיקה
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test




def normalize_data(X):
    """
    נרמול נתונים
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8), mean, std


def run_simple_test():
    """
    הרצת בדיקה משופרת על נתונים פשוטים
    """
    print("יוצר ומנרמל נתוני בדיקה...")
    X_train, y_train, X_test, y_test = create_simple_dataset()

    # נרמול הנתונים
    X_train_norm, mean_train, std_train = normalize_data(X_train)
    X_test_norm = (X_test - mean_train) / (std_train + 1e-8)

    # הגדרת ארכיטקטורה עם validation של מימדים
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    layer_dims = [input_dim, 64, 32, 16, output_dim]

    for i in range(len(layer_dims) - 1):
        if layer_dims[i + 1] < layer_dims[i] // 2:
            print(f"אזהרה: ירידה חדה במימדים בין שכבה {i} לשכבה {i + 1}")

    # יצירת המודל עם פרמטרים מכווננים
    model = NeuralNetwork(
        layer_dims=layer_dims,
        activation='relu',  # שינוי לrelu שעובד טוב יותר לרוב
        regularization='l2',
        reg_lambda=0.01
    )

    # בדיקת גרדיאנט על מדגם קטן
    print("\nמבצע בדיקת גרדיאנט...")
    gradient_sample_size = min(5, X_train.shape[0])
    small_X = X_train_norm[:gradient_sample_size]
    small_y = y_train[:gradient_sample_size]
    model.gradient_check(small_X, small_y, epsilon=1e-7, tolerance=1e-5)

    # הגדרת האופטימייזר עם פרמטרים מכווננים
    optimizer = SGDOptimizer(
        learning_rate=0.001,  # קצב למידה קטן יותר ליציבות
        momentum=0.9
    )

    # הגדרת המאמן
    trainer = NetworkTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=min(32, X_train.shape[0] // 10),  # הגבלת גודל ה-batch
        n_epochs=200,
        verbose=True
    )

    print("\nמתחיל אימון...")
    history = trainer.train(X_train_norm, y_train, X_test_norm, y_test)

    # הצגת תוצאות
    plot_training_results(history, model, X_train_norm, y_train, X_test_norm, y_test)

if __name__ == "__main__":
    run_simple_test()
# main.py
import numpy as np
from src.models.neural_network import NeuralNetwork
from src.models.resnet import ResNet
from src.optimizers.sgd import SGDOptimizer
from src.utils.trainer import NetworkTrainer
from src.utils.experiment import ExperimentRunner
from src.utils.data_loader import DataLoader


def run_network_experiments(dataset_name):
    """
    הרצת ניסויים עם ארכיטקטורות שונות
    """
    print(f"\nRunning experiments on {dataset_name} dataset")

    # טעינת נתונים
    loader = DataLoader(f"data/{dataset_name}.mat")
    (X_train, y_train), (X_val, y_val) = loader.get_data()

    # נרמול הנתונים
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # הגדרת הניסויים
    runner = ExperimentRunner(X_train, y_train, X_val, y_val)

    # ניסוי 1: רשת רגילה רדודה
    runner.run_experiment(
        "standard",
        [X_train.shape[1], 64, len(np.unique(y_train))],
        "Shallow Standard Network",
        learning_rate=0.01,
        batch_size=32,
        n_epochs=50
    )

    # ניסוי 2: רשת רגילה עמוקה
    runner.run_experiment(
        "standard",
        [X_train.shape[1], 64, 32, 16, len(np.unique(y_train))],
        "Deep Standard Network",
        learning_rate=0.01,
        batch_size=32,
        n_epochs=50
    )

    # ניסוי 3: ResNet רדודה
    runner.run_experiment(
        "resnet",
        [X_train.shape[1], 64, len(np.unique(y_train))],
        "Shallow ResNet",
        learning_rate=0.01,
        batch_size=32,
        n_epochs=50
    )

    # ניסוי 4: ResNet עמוקה
    runner.run_experiment(
        "resnet",
        [X_train.shape[1], 64, 32, 16, len(np.unique(y_train))],
        "Deep ResNet",
        learning_rate=0.01,
        batch_size=32,
        n_epochs=50
    )

    # השוואת התוצאות
    runner.compare_results()


if __name__ == "__main__":
    datasets = ["SwissRoll", "Peaks", "GMM"]

    for dataset in datasets:
        run_network_experiments(dataset)
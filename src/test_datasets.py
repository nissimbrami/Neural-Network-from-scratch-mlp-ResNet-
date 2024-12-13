import os
import numpy as np
import matplotlib.pyplot as plt
from models.neural_network import NeuralNetwork
from optimizers.sgd import SGDOptimizer
from utils.trainer import NetworkTrainer
from utils.data_loader import DataLoader
from utils.visualization import plot_data_and_decision_boundary, plot_training_results

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def normalize_data(X_train, X_test):
    
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)
    return X_train_norm, X_test_norm


def run_swiss_roll_test():
 
    print("\n=== תחילת בדיקת SwissRoll ===")

    print("\nשלב 1: טעינת נתונים...")
    data_path = os.path.join(DATA_DIR, 'SwissRollData.mat')
    print(f"נתיב הקובץ: {data_path}")

    data_loader = DataLoader(data_path)
    (X_train, y_train), (X_test, y_test) = data_loader.get_data()

    print("\nבדיקת צורת הנתונים הגולמיים:")
    print(f"X_train shape: {X_train.shape}, type: {type(X_train)}")
    print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
    print(f"X_test shape: {X_test.shape}, type: {type(X_test)}")
    print(f"y_test shape: {y_test.shape}, type: {type(y_test)}")
    print(f"ערכים ייחודיים ב-y_train: {np.unique(y_train)}")
    print(f"ערכים ייחודיים ב-y_test: {np.unique(y_test)}")

    print("\nשלב 2: נרמול נתונים...")
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)

    print("\nבדיקת צורת הנתונים אחרי נרמול:")
    print(f"X_train_norm shape: {X_train_norm.shape}, type: {type(X_train_norm)}")
    print(f"X_test_norm shape: {X_test_norm.shape}, type: {type(X_test_norm)}")
    print(f"טווח ערכים X_train_norm: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")

    print("\nשלב 3: הגדרת ארכיטקטורת הרשת...")
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    layer_dims = [input_dim, 32, 32, 32, 16, output_dim]
    print(f"ארכיטקטורת הרשת: {layer_dims}")

    print("\nשלב 4: יצירת המודל...")
    model = NeuralNetwork(
        layer_dims=layer_dims,
        activation='relu',
        regularization='l2',
        reg_lambda=0.0001
    )

    print("\nשלב 5: בדיקת גרדיאנט...")
    gradient_sample_size = min(5, X_train.shape[0])
    small_X = X_train_norm[:gradient_sample_size]
    small_y = y_train[:gradient_sample_size]
    print(f"גודל מדגם לבדיקת גרדיאנט: {gradient_sample_size}")
    print(f"צורת X לבדיקה: {small_X.shape}")
    print(f"צורת y לבדיקה: {small_y.shape}")
    model.gradient_check(small_X, small_y)

    print("\nשלב 6: הגדרת האופטימייזר...")
    optimizer = SGDOptimizer(
        learning_rate=0.005,
        momentum=0.95
    )

    print("\nשלב 7: הגדרת המאמן...")
    trainer = NetworkTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=64,
        n_epochs=200,
        verbose=True
    )

    print("\nשלב 8: תחילת אימון...")
    history = trainer.train(X_train_norm, y_train, X_test_norm, y_test)

    print("\nשלב 9: הצגת תוצאות...")
    plot_training_results(history, model, X_train_norm, y_train, X_test_norm, y_test)

    return model, history


def run_peaks_test():
  
    print("\n=== תחילת בדיקת Peaks ===")

    print("\nשלב 1: טעינת נתונים...")
    data_path = os.path.join(DATA_DIR, 'PeaksData.mat')
    print(f"נתיב הקובץ: {data_path}")

    data_loader = DataLoader(data_path)
    (X_train, y_train), (X_test, y_test) = data_loader.get_data()

    print("\nשלב 2: נרמול נתונים...")
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)

    input_dim = X_train.shape[1]  # 2
    output_dim = len(np.unique(y_train))  # 5
    layer_dims = [input_dim, 64, 128, 64, output_dim]
    print(f"ארכיטקטורת הרשת: {layer_dims}")

    model = NeuralNetwork(
        layer_dims=layer_dims,
        activation='relu',
        regularization='l2',
        reg_lambda=0.001
    )

    print("\nשלב 3: בדיקת גרדיאנט...")
    gradient_sample_size = min(5, X_train.shape[0])
    small_X = X_train_norm[:gradient_sample_size]
    small_y = y_train[:gradient_sample_size]
    model.gradient_check(small_X, small_y)

    optimizer = SGDOptimizer(
        learning_rate=0.005,
        momentum=0.9
    )

    trainer = NetworkTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=64,
        n_epochs=200,
        verbose=True
    )

    print("\nמתחיל אימון...")
    history = trainer.train(X_train_norm, y_train, X_test_norm, y_test)

    plot_training_results(history, model, X_train_norm, y_train, X_test_norm, y_test)

    return model, history


def run_gmm_test():
  
    print("\n=== בדיקת GMM (5D) ===")
    data_path = os.path.join(DATA_DIR, 'GMMData.mat')
    print(f"נתיב הקובץ: {data_path}")

    data_loader = DataLoader(data_path)
    (X_train, y_train), (X_test, y_test) = data_loader.get_data()

    print("\nמידע על הנתונים:")
    print(f"מספר דגימות אימון: {X_train.shape[0]}")
    print(f"מספר דגימות בדיקה: {X_test.shape[0]}")
    print(f"מימד הקלט: {X_train.shape[1]}")
    print(f"מספר מחלקות: {len(np.unique(y_train))}")

    X_train_norm, X_test_norm = normalize_data(X_train, X_test)

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    layer_dims = [input_dim, 128, 256, 128, output_dim]

    model = NeuralNetwork(
        layer_dims=layer_dims,
        activation='relu',
        regularization='l2',
        reg_lambda=0.001
    )

    print("\nמבצע בדיקת גרדיאנט...")
    gradient_sample_size = min(5, X_train.shape[0])
    small_X = X_train_norm[:gradient_sample_size]
    small_y = y_train[:gradient_sample_size]
    model.gradient_check(small_X, small_y)

    optimizer = SGDOptimizer(learning_rate=0.005, momentum=0.9)

    trainer = NetworkTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=64,
        n_epochs=200,
        verbose=True
    )

    print("\nמתחיל אימון...")
    history = trainer.train(X_train_norm, y_train, X_test_norm, y_test)

    plot_training_results(history, model, X_train_norm, y_train, X_test_norm, y_test)

    return model, history


def main():
  
    try:
        run_swiss_roll_test()
        run_peaks_test()
        run_gmm_test()
    except Exception as e:
        print(f"שגיאה בהרצת הניסויים: {str(e)}")
        raise


if __name__ == "__main__":
    main()

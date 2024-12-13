import os
import numpy as np
import matplotlib.pyplot as plt
from src.models.resnet import ResNet
from src.optimizers.sgd import SGDOptimizer
from src.utils.data_loader import DataLoader
from src.utils.visualization import plot_data_and_decision_boundary
from src.utils.trainer import NetworkTrainer


def get_data_path():
    """מציאת נתיב נכון לקבצי הנתונים"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data')


def normalize_data(X_train, X_test):
    """נרמול הנתונים"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


def test_resnet_on_dataset(dataset_name):
    """
    בדיקת ResNet על דאטאסט ספציפי עם הגדרות מותאמות
    """
    print(f"\n=== בדיקת ResNet על דאטאסט {dataset_name} ===")

    # טעינת נתונים
    data_path = os.path.join(get_data_path(), f'{dataset_name}Data.mat')
    data_loader = DataLoader(data_path)
    (X_train, y_train), (X_test, y_test) = data_loader.get_data()
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # פרמטרים מותאמים לכל דאטאסט
    if dataset_name == 'SwissRoll':
        # תצורה חדשה ל-SwissRoll
        hidden_dim = 64  # הגדלה משמעותית של המימד הפנימי
        hidden_dim2 = 128  # מימד גדול יותר לשכבה האמצעית
        layer_dims = [input_dim, hidden_dim, hidden_dim2, hidden_dim, num_classes]
        learning_rate = 0.001
        n_epochs = 70
        reg_lambda = 0.0001
        batch_size = 32
        momentum = 0.9
        use_dropout = False  # הסרת ה-dropout
        dropout_rate = 0.2
    else:  # GMM
        # שמירה על הפרמטרים הטובים ל-GMM
        hidden_dim = 64
        layer_dims = [input_dim, hidden_dim, hidden_dim, num_classes]
        learning_rate = 0.001
        n_epochs = 100
        reg_lambda = 0.001
        batch_size = 32
        momentum = 0.9
        use_dropout = False
        dropout_rate = 0

    print(f"\nמידע על הדאטאסט והמודל:")
    print(f"מימד קלט: {input_dim}")
    print(f"מספר מחלקות: {num_classes}")
    print(f"ארכיטקטורת רשת: {layer_dims}")
    print(f"Dropout rate: {dropout_rate if use_dropout else 'None'}")
    print(f"גודל מדגם אימון: {X_train.shape[0]}")
    print(f"גודל מדגם בדיקה: {X_test.shape[0]}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Regularization lambda: {reg_lambda}")

    # יצירת המודל
    model = ResNet(
        layer_dims=layer_dims,
        activation='relu',
        regularization='l2',
        reg_lambda=reg_lambda,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate
    )

    # הגדרת האופטימייזר
    optimizer = SGDOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )

    # הגדרת המאמן
    trainer = NetworkTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=True
    )

    # אימון
    print("\nמתחיל אימון...")
    history = trainer.train(X_train_norm, y_train, X_test_norm, y_test)

    return model, history


def main():
    """הרצת הניסויים על שני הדאטאסטים"""
    try:
        print("\nמתחיל ניסויים...")

        # הרצה על SwissRoll
        print("\n=== ניסוי 1: SwissRoll ===")
        model_swiss, history_swiss = test_resnet_on_dataset('SwissRoll')

        # הרצה על GMM
        print("\n=== ניסוי 2: GMM ===")
        model_gmm, history_gmm = test_resnet_on_dataset('GMM')

        print("\nכל הניסויים הושלמו בהצלחה!")

    except Exception as e:
        print(f"\nשגיאה בהרצת הניסויים: {str(e)}")
        raise


if __name__ == "__main__":
    main()
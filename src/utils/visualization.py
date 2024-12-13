import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def plot_gmm_projections(model, X_train, y_train, X_test, y_test):
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['o', 's']  # עיגול לאימון, ריבוע לבדיקה

    dim_pairs = list(combinations(range(X_train.shape[1]), 2))
    n_pairs = len(dim_pairs)

    n_rows = int(np.ceil(n_pairs / 2))

    plt.figure(figsize=(15, 5 * n_rows))

    for idx, (dim1, dim2) in enumerate(dim_pairs, 1):
        plt.subplot(n_rows, 2, idx)

        for class_idx in range(len(colors)):
            mask = y_train == class_idx
            plt.scatter(X_train[mask, dim1], X_train[mask, dim2],
                        c=colors[class_idx], marker=markers[0],
                        label=f'Class {class_idx} (Train)',
                        alpha=0.6)

        for class_idx in range(len(colors)):
            mask = y_test == class_idx
            plt.scatter(X_test[mask, dim1], X_test[mask, dim2],
                        c=colors[class_idx], marker=markers[1],
                        label=f'Class {class_idx} (Test)',
                        alpha=0.6)

        plt.title(f'Dimensions {dim1 + 1} vs {dim2 + 1}')
        plt.xlabel(f'Dimension {dim1 + 1}')
        plt.ylabel(f'Dimension {dim2 + 1}')
        if idx == 1:  # רק בגרף הראשון נציג מקרא
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_peaks_decision_boundary(model, X_train, y_train, X_test, y_test):
 
    h = 0.02
    x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1, max(X_train[:, 0].max(), X_test[:, 0].max()) + 1
    y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1, max(X_train[:, 1].max(), X_test[:, 1].max()) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.forward_prop(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    plt.figure(figsize=(15, 6))

    plt.subplot(121)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.get_cmap('Set3', 5))
    for i in range(5):
        plt.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1],
                    c=colors[i], marker='o', label=f'Class {i} (Train)', alpha=0.8)
    plt.title("Training Data and Decision Boundary", pad=20)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.subplot(122)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.get_cmap('Set3', 5))
    for i in range(5):
        plt.scatter(X_test[y_test == i][:, 0], X_test[y_test == i][:, 1],
                    c=colors[i], marker='s', label=f'Class {i} (Test)', alpha=0.8)
    plt.title("Test Data and Decision Boundary", pad=20)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.tight_layout()
    plt.show()


def plot_data_and_decision_boundary(model, X_train, y_train, X_test, y_test, title="Decision Boundary"):

    if X_train.shape[1] == 5:  # GMM
        plot_gmm_projections(model, X_train, y_train, X_test, y_test)
    elif len(np.unique(y_train)) == 5:  # Peaks
        plot_peaks_decision_boundary(model, X_train, y_train, X_test, y_test)
    else:  # SwissRoll
        h = 0.02
        x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1, max(X_train[:, 0].max(),
                                                                             X_test[:, 0].max()) + 1
        y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1, max(X_train[:, 1].max(),
                                                                             X_test[:, 1].max()) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.forward_prop(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(15, 6))

        plt.subplot(121)
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)
        plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
                    c='blue', marker='o', label='Class 0 (Train)', alpha=0.8)
        plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                    c='red', marker='o', label='Class 1 (Train)', alpha=0.8)
        plt.title("Training Data and Decision Boundary", pad=20)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X1')
        plt.ylabel('X2')

        plt.subplot(122)
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)
        plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
                    c='blue', marker='s', label='Class 0 (Test)', alpha=0.8)
        plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
                    c='red', marker='s', label='Class 1 (Test)', alpha=0.8)
        plt.title("Test Data and Decision Boundary", pad=20)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X1')
        plt.ylabel('X2')

        plt.tight_layout()
        plt.show()


def plot_training_results(results):

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    for name, res in results.items():
        plt.plot(res['train_acc'], label=f'{name}')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(132)
    for name, res in results.items():
        plt.plot(res['val_acc'], label=f'{name}')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(133)
    for name, res in results.items():
        plt.plot(res['loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

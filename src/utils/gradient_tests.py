import numpy as np
import matplotlib.pyplot as plt


def plot_slope_test(f, x, d, title):
    """
    בדיקת גרדיאנט באמצעות חישוב שיפועים

    Args:
        f: פונקציה לבדיקה
        x: נקודת הבדיקה
        d: כיוון ההפרעה
        title: כותרת לגרף
    """
    # וקטור של גדלי צעד
    h = np.logspace(-8, 0, 20)

    # חישוב שיפועים
    slopes = []
    for step in h:
        slope = (f(x + step * d) - f(x)) / step
        slopes.append(slope)

    # הצגה בסקלה לוגריתמית
    plt.figure(figsize=(10, 6))
    plt.loglog(h, np.abs(slopes), 'bo-', label='Computed slope')
    plt.grid(True)
    plt.xlabel('Step size (h)')
    plt.ylabel('|Slope|')
    plt.title(f'Gradient Test - {title}')
    plt.legend()
    plt.show()


def least_squares_example():
    """
    דוגמה של אופטימיזציית SGD על בעיית ריבועים פחותים
    """
    # יצירת נתונים סינטטיים
    np.random.seed(42)
    X = np.random.randn(100, 2)
    w_true = np.array([2.0, -1.5])
    y = np.dot(X, w_true) + np.random.randn(100) * 0.1

    # פונקציית Loss
    def loss_fn(w):
        return np.mean((np.dot(X, w) - y) ** 2)

    # פונקציית גרדיאנט
    def grad_fn(w):
        return 2 * np.mean(X.T * (np.dot(X, w) - y)[:, np.newaxis], axis=1)

    # אופטימיזציה עם SGD
    from ..optimizers.sgd import SGDOptimizer

    optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9)
    w = np.zeros(2)
    losses = []

    for i in range(100):
        loss = loss_fn(w)
        grad = grad_fn(w)
        w = optimizer.step({'w': w}, {'dw': grad})['w']
        losses.append(loss)

    # הצגת תוצאות
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.grid(True)

    plt.subplot(122)
    plt.scatter(X[:, 0], y, label='Data')
    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    plt.plot(x_line, w[0] * x_line + w[1] * np.ones_like(x_line),
             'r-', label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Fitted Line')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return w, w_true, losses


def jacobian_test_layer(layer, input_data, epsilon=1e-7):
    """
    בדיקת Jacobian לשכבה בודדת
    """
    output = layer.forward(input_data)

    # יצירת הפרעה רנדומלית
    d = np.random.randn(*input_data.shape)
    d = d / np.linalg.norm(d)

    # חישוב הJacobian באופן נומרי
    h = np.logspace(-8, -1, 20)
    slopes = []

    for step in h:
        output_plus = layer.forward(input_data + step * d)
        slope = np.linalg.norm(output_plus - output) / step
        slopes.append(slope)

    # הצגת תוצאות
    plt.figure(figsize=(10, 6))
    plt.loglog(h, slopes, 'bo-', label='Numerical')
    plt.grid(True)
    plt.xlabel('Step size (h)')
    plt.ylabel('|J*d|')
    plt.title('Jacobian Test')
    plt.legend()
    plt.show()
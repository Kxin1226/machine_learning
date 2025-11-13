"""
SVR experiments on the Wine Quality (white) dataset.

Loads winequality-white.csv, uses 20% of data as working set, then splits 80/20 for train/test.
Runs SVR with kernels linear, rbf, poly, sigmoid and computes MSE. Saves prediction plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def cubic_kernel(X, Y):
    return (np.dot(X, Y.T) + 1) ** 3


def poly_kernel(X, Y, degree=3, coef0=1):
    return (np.dot(X, Y.T) + coef0) ** degree


def gaussian_kernel(X, Y, gamma=0.5):
    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Y_norm = np.sum(Y ** 2, axis=1)[None, :]
    K = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * K)


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_wine(path):
    # try to auto-detect separator; fall back to common separators
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception:
        for sep in [',', ';']:
            try:
                df = pd.read_csv(path, sep=sep)
                break
            except Exception:
                df = None
    if df is None:
        raise RuntimeError(f"Could not read CSV file: {path}")
    return df


def plot_pred_true(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    m = min(y_true.min(), y_pred.min())
    M = max(y_true.max(), y_pred.max())
    plt.plot([m, M], [m, M], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def main():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "winequality-white.csv")
    df = load_wine(DATA_PATH)

    # features and target
    if 'quality' in df.columns:
        y = df['quality'].values
        X = df.drop(columns=['quality']).values
    else:
        # fallback: assume last column is target
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values

    # 使用20%的数据作为工作集
    _, working = train_test_split(np.hstack([X, y.reshape(-1, 1)]), test_size=0.2, random_state=42, shuffle=True)
    Xw = working[:, :-1]
    yw = working[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(Xw, yw, test_size=0.2, random_state=1)

    # 标准化特征（对 SVR 很重要）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']

    for kernel in kernels:
        svr = SVR(kernel=kernel)
        svr.fit(X_train_s, y_train)
        y_pred = svr.predict(X_test_s)
        mse = mean_squared_error(y_test, y_pred)
        title = f"SVR kernel={kernel} MSE={mse:.4f}"
        fname = f"svr_{kernel}.png"
        plot_pred_true(y_test, y_pred, title, fname)
        print(f"Kernel: {kernel}, MSE: {mse:.4f}")

    # 自定义核测试（使用已标准化的数据）
    custom_kernels = {
        'cubic': lambda X, Y: cubic_kernel(X, Y),
        'poly3': lambda X, Y: poly_kernel(X, Y, degree=3, coef0=1),
        'gaussian': lambda X, Y: gaussian_kernel(X, Y, gamma=0.5)
    }

    for name, ker in custom_kernels.items():
        svr = SVR(kernel=ker)
        svr.fit(X_train_s, y_train)
        y_pred = svr.predict(X_test_s)
        mse = mean_squared_error(y_test, y_pred)
        title = f"SVR custom={name} MSE={mse:.4f}"
        fname = f"svr_custom_{name}.png"
        plot_pred_true(y_test, y_pred, title, fname)
        print(f"Custom kernel: {name}, MSE: {mse:.4f}")


if __name__ == '__main__':
    main()

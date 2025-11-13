"""
SVC experiments on the Banknote Authentication dataset.

Creates plots for different kernel SVM classifiers using pairs of features,
tests custom kernels (cubic, polynomial, gaussian), and saves figures to ./outputs.
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_banknote(path):
    # dataset has no header, 5 columns: var, skew, curtosis, entropy, class
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def cubic_kernel(X, Y):
    return (np.dot(X, Y.T) + 1) ** 3


def poly_kernel(X, Y, degree=3, coef0=1):
    return (np.dot(X, Y.T) + coef0) ** degree


def gaussian_kernel(X, Y, gamma=0.5):
    # returns kernel matrix
    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Y_norm = np.sum(Y ** 2, axis=1)[None, :]
    K = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * K)


def plot_decision_boundary(clf, X, y, feature_names, title, filename, plot_step=0.05):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    grid = np.c_[xx.ravel(), yy.ravel()]

    try:
        Z = clf.predict(grid)
    except Exception:
        # some custom kernels or precomputed may require special handling
        Z = np.array([clf.predict(g.reshape(1, -1))[0] for g in grid])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def main():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data_banknote_authentication.txt")
    X, y = load_banknote(DATA_PATH)

    # 随机选取20%作为工作集
    _, working = train_test_split(np.hstack([X, y.reshape(-1, 1)]), test_size=0.2, random_state=42, shuffle=True)
    Xw = working[:, :-1]
    yw = working[:, -1].astype(int)

    # 再划分为 80% 训练集 20% 测试集
    X_train, X_test, y_train, y_test = train_test_split(Xw, yw, test_size=0.2, random_state=1)

    feature_indices = [(0, 1), (0, 2), (1, 2)]
    feature_names = ["variance", "skewness", "curtosis", "entropy"]

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']

    for (i, j) in feature_indices:
        Xtr = X_train[:, [i, j]]
        Xte = X_test[:, [i, j]]

        for kernel in kernels:
            clf = SVC(kernel=kernel, gamma='scale')
            clf.fit(Xtr, y_train)
            ypred = clf.predict(Xte)
            acc = accuracy_score(y_test, ypred)
            title = f"SVC kernel={kernel} features={i},{j} acc={acc:.3f}"
            fname = f"svc_{i}_{j}_{kernel}.png"
            plot_decision_boundary(clf, Xtr, y_train, (feature_names[i], feature_names[j]), title, fname)

    # 自定义核（作为可调用 kernel）示例
    custom_kernels = {
        'cubic': lambda X, Y: cubic_kernel(X, Y),
        'poly3': lambda X, Y: poly_kernel(X, Y, degree=3, coef0=1),
        'gaussian': lambda X, Y: gaussian_kernel(X, Y, gamma=0.5)
    }

    # 测试第一个特征对上的自定义核
    i, j = feature_indices[0]
    Xtr = X_train[:, [i, j]]
    Xte = X_test[:, [i, j]]

    for name, ker in custom_kernels.items():
        clf = SVC(kernel=ker)
        clf.fit(Xtr, y_train)
        ypred = clf.predict(Xte)
        acc = accuracy_score(y_test, ypred)
        title = f"SVC custom={name} features={i},{j} acc={acc:.3f}"
        fname = f"svc_custom_{name}_{i}_{j}.png"
        plot_decision_boundary(clf, Xtr, y_train, (feature_names[i], feature_names[j]), title, fname)

    # Gram 矩阵示例（在整个工作集上，演示 precomputed）
    X_full = Xw[:, [i, j]]
    y_full = yw
    K = cubic_kernel(X_full, X_full)
    clf_pre = SVC(kernel='precomputed')
    clf_pre.fit(K, y_full)
    # 将训练点作为网格进行可视化（不使用独立测试集）
    # 为决策边界，直接在原点上画出预测（注意：precomputed 需要特殊处理）
    def predict_grid_for_precomputed(clf, X_train, grid):
        K_test = cubic_kernel(grid, X_train)
        return clf.predict(K_test)

    # small grid using feature ranges
    x_min, x_max = X_full[:, 0].min() - 1, X_full[:, 0].max() + 1
    y_min, y_max = X_full[:, 1].min() - 1, X_full[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_grid_for_precomputed(clf_pre, X_full, grid).reshape(xx.shape)
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title('SVC precomputed cubic on full working set')
    path = os.path.join(OUT_DIR, 'svc_precomputed_cubic.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


if __name__ == '__main__':
    main()

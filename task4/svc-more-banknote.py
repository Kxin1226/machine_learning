"""
Extension experiments using more features on the Banknote dataset.

Performs pairwise decision boundary plots across all feature pairs, runs 10-fold CV
to select the best kernel (linear, rbf, poly, sigmoid) and reports precision/recall/F1.
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_banknote(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def plot_pair(X, y, i, j, clf, feature_names, fname, plot_step=0.06):
    x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
    y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # train clf on these two features
    X2 = X[:, [i, j]]
    try:
        clf.fit(X2, y)
        Z = clf.predict(grid)
    except Exception:
        Z = np.array([clf.predict(g.reshape(1, -1))[0] for g in grid])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.7)
    plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=20)
    plt.xlabel(feature_names[i])
    plt.ylabel(feature_names[j])
    plt.title(f"features {i}-{j}")
    path = os.path.join(OUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def main():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data_banknote_authentication.txt")
    X, y = load_banknote(DATA_PATH)

    # 使用20%作为工作集
    _, working = train_test_split(np.hstack([X, y.reshape(-1, 1)]), test_size=0.2, random_state=42, shuffle=True)
    Xw = working[:, :-1]
    yw = working[:, -1].astype(int)

    # 再划分为训练/测试
    X_train, X_test, y_train, y_test = train_test_split(Xw, yw, test_size=0.2, random_state=1)

    feature_names = ["variance", "skewness", "curtosis", "entropy"]

    # 绘制所有特征对的决策边界（使用 rbf 作为示例）
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    for (i, j) in pairs:
        clf = SVC(kernel='rbf', gamma='scale')
        plot_pair(X_train, y_train, i, j, clf, feature_names, f"pair_{i}_{j}_rbf.png")

    # 10 折交叉验证选择最佳核
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    best = None
    best_score = -1
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for kernel in kernels:
        clf = SVC(kernel=kernel, gamma='scale')
        scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='f1')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"kernel={kernel} CV F1 mean={mean_score:.4f} std={std_score:.4f} per-fold={np.round(scores,4)}")
        if mean_score > best_score:
            best_score = mean_score
            best = kernel

    print(f"Best kernel by 10-fold CV (F1): {best} (F1={best_score:.4f})")

    # 对每个内置核，在训练集上训练并在测试集上评估（输出每个模型的指标）
    print('\n=== Evaluation on test set for each kernel ===')
    print(f"Test size: {len(y_test)}")
    try:
        import pandas as _pd
        print('Test set class distribution:')
        print(_pd.Series(y_test).value_counts().to_string())
    except Exception:
        pass

    # prepare metrics output
    metrics_lines = []
    metrics_lines.append('=== Evaluation on test set for each kernel ===')
    metrics_lines.append(f'Test size: {len(y_test)}')
    try:
        import pandas as _pd
        metrics_lines.append('Test set class distribution:')
        metrics_lines.extend([f"{k}: {v}" for k, v in _pd.Series(y_test).value_counts().items()])
    except Exception:
        pass

    for kernel in kernels:
        clf = SVC(kernel=kernel, gamma='scale')
        clf.fit(X_train, y_train)
        y_pred_k = clf.predict(X_test)
        acc_k = accuracy_score(y_test, y_pred_k)
        prec_k = precision_score(y_test, y_pred_k)
        rec_k = recall_score(y_test, y_pred_k)
        f1_k = f1_score(y_test, y_pred_k)
        print(f"\nKernel: {kernel}")
        print(f"Accuracy: {acc_k:.4f}, Precision: {prec_k:.4f}, Recall: {rec_k:.4f}, F1: {f1_k:.4f}")
        print('Confusion matrix:')
        cm = confusion_matrix(y_test, y_pred_k)
        print(cm)

        # append to metrics_lines
        metrics_lines.append(f'')
        metrics_lines.append(f'Kernel: {kernel}')
        metrics_lines.append(f'Accuracy: {acc_k:.4f}, Precision: {prec_k:.4f}, Recall: {rec_k:.4f}, F1: {f1_k:.4f}')
        metrics_lines.append('Confusion matrix:')
        for row in cm:
            metrics_lines.append(' '.join(map(str, row)))

    # 另外也输出每模型的 classification report 简短版并保存
    metrics_lines.append('')
    metrics_lines.append('=== Classification reports (short) ===')
    for kernel in kernels:
        clf = SVC(kernel=kernel, gamma='scale')
        clf.fit(X_train, y_train)
        y_pred_k = clf.predict(X_test)
        print(f"\nKernel: {kernel}")
        crep = classification_report(y_test, y_pred_k, digits=4)
        print(crep)
        metrics_lines.append(f'\nKernel: {kernel}')
        metrics_lines.extend(crep.splitlines())

    # write metrics to file
    metrics_path = os.path.join(OUT_DIR, 'metrics.txt')
    try:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metrics_lines))
        print(f"Saved metrics to {metrics_path}")
    except Exception as e:
        print('Failed to write metrics:', e)


if __name__ == '__main__':
    main()

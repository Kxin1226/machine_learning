# section5_nearest_neighbour.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from math import sqrt

def confusion_counts(y_true, y_pred, pos_label):
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    tn = int(np.sum((y_pred != pos_label) & (y_true != pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    return tp, tn, fp, fn

def precision_recall_f1_from_counts(tp, tn, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def main():
    # 加载数据
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target
    # 约定 positive label 与前面脚本一致（malignant -> 0），但此处为了通用直接做判断
    target_names = bc.target_names
    if any('malign' in n.lower() for n in target_names):
        pos_label = 0
    else:
        pos_label = 1

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    precisions = []
    recalls = []
    f1s = []

    fold = 1
    for train_idx, test_idx in kf.split(X):
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        tp, tn, fp, fn = confusion_counts(y[test_idx], y_pred, pos_label)
        prec, rec, f1 = precision_recall_f1_from_counts(tp, tn, fp, fn)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        print(f"Fold {fold}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        fold += 1

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    n = len(precisions)

    print("\nSummary across 10 folds (K=5 NN):")
    print(f" Precision: mean={precisions.mean():.4f}, std={precisions.std(ddof=0):.4f}")
    print(f" Recall   : mean={recalls.mean():.4f}, std={recalls.std(ddof=0):.4f}")
    print(f" F1 score : mean={f1s.mean():.4f}, std={f1s.std(ddof=0):.4f}")

if __name__ == "__main__":
    main()

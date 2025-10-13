# section3_cross_validation.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold

def main():
    # 加载数据
    bc = load_breast_cancer()
    X = bc.data
    y = bc.target

    print("=== Section 3: Cross-validation ===\n")

    # 1) 90% train / 10% test 一次划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    dtree = DecisionTreeClassifier(criterion="entropy", random_state=0)
    dtree.fit(X_train, y_train)
    acc = dtree.score(X_test, y_test)
    print(f"1) Single 90/10 split accuracy: {acc:.4f}")

    # 2) 随机划分 10 次，计算平均准确率与标准差
    scores = []
    for seed in range(10):
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.1, random_state=seed)
        clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
        clf.fit(Xt, yt)
        sc = clf.score(Xv, yv)
        scores.append(sc)
    scores = np.array(scores)
    print(f"\n2) 10 random 90/10 splits accuracies:\n  {scores}")
    print(f"   mean = {scores.mean():.4f}, std = {scores.std(ddof=0):.4f}")

    # 3) sklearn cross_val_score 10-fold
    dtree_cv = DecisionTreeClassifier(criterion="entropy", random_state=0)
    cv_scores = cross_val_score(dtree_cv, X, y, cv=10)  # default scoring = accuracy
    print(f"\n3) cross_val_score (cv=10) accuracies:\n  {cv_scores}")
    print(f"   mean = {cv_scores.mean():.4f}, std = {cv_scores.std(ddof=0):.4f}")

    # 4) 手写 10-fold CV（使用 KFold 默认分割：不 shuffle）
    kf = KFold(n_splits=10, shuffle=False)
    manual_scores = []
    for train_idx, test_idx in kf.split(X):
        clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        manual_scores.append(clf.score(X[test_idx], y[test_idx]))
    manual_scores = np.array(manual_scores)
    print(f"\n4) manual 10-fold CV accuracies:\n  {manual_scores}")
    print(f"   mean = {manual_scores.mean():.4f}, std = {manual_scores.std(ddof=0):.4f}")

    # 简短比较
    print("\nSummary comparison:")
    print(f" - 10 random 90/10 splits mean: {scores.mean():.4f}, std: {scores.std(ddof=0):.4f}")
    print(f" - cross_val_score (cv=10) mean: {cv_scores.mean():.4f}, std: {cv_scores.std(ddof=0):.4f}")
    print(f" - manual 10-fold mean: {manual_scores.mean():.4f}, std: {manual_scores.std(ddof=0):.4f}")

if __name__ == "__main__":
    main()

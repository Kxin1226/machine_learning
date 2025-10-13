# section4_metrics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

def determine_positive_label(target_names):
    # 如果 target_names 含 'malignant'，就把 malignant 作为 positive（医学场景）；
    # 否则默认选最后一个类作为 positive。
    names_lower = [n.lower() for n in target_names]
    if any('malign' in n for n in names_lower):
        # malignant 在 sklearn 中通常是 index 0
        return 0
    return len(target_names) - 1

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
    target_names = bc.target_names

    pos_label = determine_positive_label(target_names)
    other_label = [c for c in np.unique(y) if c != pos_label][0]
    print("Positive label chosen as:", pos_label, "(", target_names[pos_label], ")\n")

    # 1) 用 90%/10% 划分训练一个决策树，用于混淆矩阵计算
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    dtree = DecisionTreeClassifier(criterion="entropy", random_state=0)
    dtree.fit(X_train, y_train)

    # 手工逐样本计算混淆矩阵
    y_pred = dtree.predict(X_test)
    tp, tn, fp, fn = confusion_counts(y_test, y_pred, pos_label)
    print("Confusion matrix counts (w.r.t. positive label):")
    print(f" TP = {tp}, TN = {tn}, FP = {fp}, FN = {fn}")
    prec, rec, f1 = precision_recall_f1_from_counts(tp, tn, fp, fn)
    print(f" Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}\n")

    # 2) 在手写 10-fold CV 中计算每折的 precision/recall，并输出均值与标准误
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    precisions = []
    recalls = []
    for train_idx, test_idx in kf.split(X):
        clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        y_pred_fold = clf.predict(X[test_idx])
        tp_f, tn_f, fp_f, fn_f = confusion_counts(y[test_idx], y_pred_fold, pos_label)
        p_f, r_f, _ = precision_recall_f1_from_counts(tp_f, tn_f, fp_f, fn_f)
        precisions.append(p_f)
        recalls.append(r_f)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    n = len(precisions)
    prec_mean, prec_std, prec_se = precisions.mean(), precisions.std(ddof=0), precisions.std(ddof=0)/np.sqrt(n)
    rec_mean, rec_std, rec_se = recalls.mean(), recalls.std(ddof=0), recalls.std(ddof=0)/np.sqrt(n)
    print("10-fold CV (manual) Precision: per-fold =", np.round(precisions,4))
    print(f"  mean = {prec_mean:.4f}, std = {prec_std:.4f}, standard error = {prec_se:.4f}")
    print("10-fold CV (manual) Recall: per-fold =", np.round(recalls,4))
    print(f"  mean = {rec_mean:.4f}, std = {rec_std:.4f}, standard error = {rec_se:.4f}\n")

    # 3) ROC 曲线：使用 k-NN (k=5) + predict_proba，手工遍历阈值计算 TPR/FPR 并绘图
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)  # 使用前面 90% 的训练集
    # 找到 positive 类在 predict_proba 输出列中的索引
    pos_index = list(knn.classes_).index(pos_label)
    proba_pos = knn.predict_proba(X_test)[:, pos_index]

    # 通过不同阈值计算 TPR/FPR
    thresholds = np.linspace(0.0, 1.0, 101)
    tprs = []
    fprs = []
    for thr in thresholds:
        y_pred_thr = np.where(proba_pos >= thr, pos_label, other_label)
        tp_t, tn_t, fp_t, fn_t = confusion_counts(y_test, y_pred_thr, pos_label)
        tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    # 计算 AUC（使用概率和 sklearn 的函数作为补充）
    try:
        auc = roc_auc_score((y_test == pos_label).astype(int), proba_pos)
    except Exception:
        auc = None

    # 绘图
    plt.figure(figsize=(6,6))
    plt.plot(fprs, tprs, marker='o', linestyle='-', markersize=3, label=f'KNN ROC (AUC={auc:.4f})' if auc is not None else 'KNN ROC')
    plt.plot([0,1],[0,1],'k--', label='random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC curve (KNN, manual thresholds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("ROC AUC (sklearn roc_auc_score on predicted probabilities):", None if auc is None else f"{auc:.4f}")

if __name__ == "__main__":
    main()

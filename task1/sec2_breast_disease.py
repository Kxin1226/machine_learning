import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. 加载数据
bc = load_breast_cancer()
X = bc.data
y = bc.target
feature_names = bc.feature_names
class_names = bc.target_names

# 2. 用所有特征训练一次，获取特征重要性
clf_all = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0)
clf_all.fit(X, y)
importances = clf_all.feature_importances_

# 3. 选出前3个最重要特征
indices = np.argsort(importances)[::-1]
top3_idx = indices[:3]
top3_features = [feature_names[i] for i in top3_idx]

print("Top 3 features:", top3_features)

# 4. 用前3个特征重新训练决策树
X_top3 = X[:, top3_idx]
clf_top3 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
clf_top3.fit(X_top3, y)

# 5. 可视化决策树（字体缩小）
plt.figure(figsize=(16, 8))
plot_tree(clf_top3,
          feature_names=top3_features,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=8)   # 字体调小
plt.show()
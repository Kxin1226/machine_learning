import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ========== 1. 构造西瓜数据集3.0 ==========
data = {
    '色泽': ['青绿','乌黑','乌黑','青绿','浅白','青绿','乌黑','乌黑','乌黑','青绿',
           '浅白','青绿','浅白','青绿','浅白','浅白','乌黑'],
    '根蒂': ['蜷缩','蜷缩','蜷缩','蜷缩','蜷缩','稍蜷','稍蜷','稍蜷','稍蜷','硬挺',
           '硬挺','硬挺','硬挺','稍蜷','稍蜷','蜷缩','稍蜷'],
    '敲声': ['浊响','沉闷','浊响','浊响','清脆','浊响','浊响','沉闷','清脆','沉闷',
           '浊响','清脆','清脆','沉闷','沉闷','沉闷','浊响'],
    '纹理': ['清晰','清晰','清晰','清晰','模糊','清晰','稍糊','清晰','模糊','稍糊',
           '模糊','稍糊','清晰','稍糊','稍糊','模糊','稍糊'],
    '脐部': ['凹陷','凹陷','凹陷','凹陷','平坦','凹陷','稍凹','凹陷','平坦','稍凹',
           '平坦','稍凹','平坦','稍凹','稍凹','平坦','稍凹'],
    '触感': ['硬滑','硬滑','硬滑','硬滑','软粘','硬滑','软粘','硬滑','硬滑','软粘',
           '硬滑','硬滑','硬滑','硬滑','软粘','软粘','硬滑'],
    '密度': [0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,
           0.245,0.343,0.639,0.657,0.593,0.719,0.359],
    '含糖率': [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,
             0.057,0.099,0.161,0.198,0.042,0.103,0.188],
    '好瓜': ['是','是','是','是','是','是','是','是','否','否',
           '否','否','否','否','否','否','否']
}
df = pd.DataFrame(data)

# ========== 2. 编码处理 ==========
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop('好瓜', axis=1)
y = df['好瓜']

# ========== 3. 定义基于逻辑回归的划分选择函数 ==========
def logistic_split_score(X, y, feature):
    """计算某个特征的平均对数损失(越小越好)"""
    unique_vals = np.unique(X[feature])
    total_loss = 0
    for val in unique_vals:
        mask = (X[feature] == val)
        if len(np.unique(y[mask])) == 1:
            continue
        X_sub = X.loc[mask].drop(columns=[feature])
        y_sub = y[mask]
        model = LogisticRegression(max_iter=500)
        model.fit(X_sub, y_sub)
        prob = model.predict_proba(X_sub)
        loss = -np.mean(y_sub*np.log(prob[:,1]) + (1 - y_sub)*np.log(prob[:,0]))
        total_loss += loss * len(y_sub)
    return total_loss / len(y)

# 计算各特征划分得分
scores = {f: logistic_split_score(X, y, f) for f in X.columns}
best_feature = min(scores, key=scores.get)

print("各特征逻辑回归划分得分（越小越好）:")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")
print(f"\n最优划分特征: {best_feature}")

# ========== 4. 基于此特征训练决策树 ==========
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X, y)

# ========== 5. 中文绘图设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 绘制决策树
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=X.columns,
          class_names=['坏瓜','好瓜'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("基于对率回归划分准则的决策树", fontsize=14)
plt.show()

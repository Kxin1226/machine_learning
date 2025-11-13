import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# === 1. 构造西瓜数据集3.0 ===
data = [
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '是'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '是'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '是'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '是'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '是'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '是'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '是'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '软粘', 0.437, 0.211, '是'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '平坦', '硬滑', 0.666, 0.091, '否'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '否'],
    ['浅白', '硬挺', '模糊', '清晰', '平坦', '硬滑', 0.245, 0.057, '否'],
    ['浅白', '蜷缩', '模糊', '稍糊', '平坦', '软粘', 0.343, 0.099, '否'],
    ['青绿', '稍蜷', '清脆', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '否'],
    ['浅白', '稍蜷', '浊响', '模糊', '凹陷', '硬滑', 0.657, 0.198, '否'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.593, 0.042, '否'],
    ['浅白', '蜷缩', '浊响', '模糊', '稍凹', '硬滑', 0.719, 0.103, '否'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '否']
]
columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']
df = pd.DataFrame(data, columns=columns)

# === 2. 计算信息熵 ===
def entropy(data):
    labels = data['好瓜']
    probs = labels.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in probs)

# === 3. 计算信息增益 ===
def info_gain(data, feature):
    base_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += len(subset) / len(data) * entropy(subset)
    return base_entropy - weighted_entropy

# === 4. 选择最佳划分特征 ===
def best_feature(data):
    features = data.columns[:-1]  # 去掉标签列
    gains = {f: info_gain(data, f) for f in features if data[f].dtype == 'O'}
    return max(gains, key=gains.get)

# === 5. 递归构建决策树 ===
def build_tree(data):
    labels = data['好瓜']
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if len(data.columns) == 1:
        return labels.value_counts().idxmax()
    
    best = best_feature(data)
    tree = {best: {}}
    for v in data[best].unique():
        subset = data[data[best] == v].drop(columns=[best])
        tree[best][v] = build_tree(subset)
    return tree

# === 6. 构建决策树 ===
tree = build_tree(df)
print("决策树结构：")
print(tree)

# === 7. Matplotlib可视化 ===
def plot_tree(tree, parent_name, pos=None, level=0, width=1.0, dx=1.0, dy=1.5):
    if pos is None:
        pos = (0, 0)
    feature = list(tree.keys())[0]
    values = list(tree[feature].keys())
    n = len(values)
    x0, y0 = pos
    step = width / n
    for i, v in enumerate(values):
        child = tree[feature][v]
        x_child = x0 - width / 2 + step / 2 + i * step
        y_child = y0 - dy
        plt.text(x_child, y_child, f'{v}', ha='center', va='center', fontsize=10)
        plt.plot([x0, x_child], [y0 - 0.1, y_child + 0.1], 'k-')
        if isinstance(child, dict):
            plt.text(x0, y0, feature, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue'))
            plot_tree(child, feature, (x_child, y_child - 0.2),
                      level + 1, width=step * 0.8, dx=dx, dy=dy)
        else:
            plt.text(x_child, y_child - 0.2, f'{child}',
                     ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))

plt.figure(figsize=(10, 6))
plot_tree(tree, 'root')
plt.axis('off')
plt.title("基于信息增益的ID3决策树", fontsize=13)
plt.show()

# K-Means 聚类作业 (Lab03) - 完整版

## 📋 文件说明

本文件夹包含完整的 K-means 聚类作业，所有文件已按照 lab03.pdf 要求组织：

### 主要文件
- **`assignment_new.py`**: 完整的作业代码（推荐使用）
  - ✅ 按照 PDF 章节顺序组织（2→3→4→5→6）
  - ✅ 每部分开始前有清晰的标题注释
  - ✅ 关键步骤包含详细中文注释
  - ✅ 图片按章节编号（3_*.png, 5_*.png, 6_1_*.png, 6_2_*.png）
  
- **`kmeans_assignment.py`**: 之前版本的代码（也可以使用）
- **`k-means-starter.py`**: 原始的起始代码
- **`lab03-1.pdf`**: 作业说明文档

### 输出文件夹
- **`new_output/`**: 新版本代码生成的图片（推荐提交此文件夹）
- **`outputs/`**: 旧版本代码生成的图片

### 文档
- **`实验报告模板.md`**: 实验报告模板，请填写实验结果和分析
- **`README.md`**: 本说明文件
- **`requirements.txt`**: Python 依赖清单

---

## 🎯 作业完成情况

### ✅ 已完成的所有要求

#### 2. K-means for the iris data
- ✅ 实现完整的 K-means 算法类
- ✅ 使用随机初始化选择聚类中心
- ✅ 在 Iris 数据集上训练（使用特征索引 1 和 3）
- ✅ 输出训练时间、迭代次数、inertia、聚类中心

#### 3. Plotting results
- ✅ 绘制真实标签 vs K-means 结果的对比图
- ✅ 图片保存为 `3_iris_kmeans_vs_true.png`

#### 4. Measuring performance
- ✅ 计算 Adjusted Rand Index (ARI)
- ✅ 额外计算 Silhouette Score
- ✅ 在终端输出所有性能指标

#### 5. Comparing with scikit-learn
- ✅ 使用 sklearn.cluster.KMeans 进行对比
- ✅ 生成对比表格（时间、迭代次数、inertia、ARI）
- ✅ 绘制三方对比图（真实标签、自定义、sklearn）
- ✅ 图片保存为 `5_comparison_custom_vs_sklearn.png`

#### 6.1 K-means++ initialization
- ✅ 实现 K-means++ 初始化方法
- ✅ 使用 make_blobs 生成 2 维测试数据
- ✅ 对比随机初始化 vs K-means++ 的性能
- ✅ 绘制对比图，保存为 `6_1_kmeans_plusplus_comparison.png`

#### 6.2 Generalize to any number of clusters
- ✅ 代码泛化，支持任意簇数量（n_clusters 参数）
- ✅ 测试 3、5、6 个簇的情况
- ✅ 使用 make_blobs 生成 2 维测试数据
- ✅ 每个测试生成对比图（算法结果 vs 真实标签）
- ✅ 图片保存为 `6_2_generalized_*.png`

---

## 🚀 运行方法

### 方式 1: 运行新版本代码（推荐）

```powershell
# 在 task3 目录下运行
python assignment_new.py
```

**输出**:
- 终端显示所有章节的执行过程和性能指标
- 在 `new_output/` 文件夹生成 6 张图片

### 方式 2: 运行旧版本代码

```powershell
python kmeans_assignment.py
```

**输出**:
- 在 `outputs/` 文件夹生成图片

### 安装依赖

如果缺少依赖包：

```powershell
pip install -r requirements.txt
```

---

## 📊 生成的图片清单

### 新版本 (new_output/)
1. **`3_iris_kmeans_vs_true.png`** - Iris 数据集：真实标签 vs K-means 结果
2. **`5_comparison_custom_vs_sklearn.png`** - 三方对比：真实标签 vs 自定义 vs sklearn
3. **`6_1_kmeans_plusplus_comparison.png`** - K-means++ vs 随机初始化对比
4. **`6_2_generalized_3_clusters.png`** - 3 个簇的泛化测试
5. **`6_2_generalized_5_clusters.png`** - 5 个簇的泛化测试
6. **`6_2_generalized_6_clusters.png`** - 6 个簇的泛化测试

---

## 📝 代码组织结构

`assignment_new.py` 按照以下顺序组织：

```
# 导入库

# ============================================================================
# 2. K-means for the iris data
# ============================================================================
# - 加载 Iris 数据集
# - 实现 KMeans 类（包含随机初始化和 K-means++）
# - 训练模型

# ============================================================================
# 3. Plotting results
# ============================================================================
# - 绘制真实标签 vs K-means 结果

# ============================================================================
# 4. Measuring performance
# ============================================================================
# - 计算 ARI、Silhouette Score 等指标

# ============================================================================
# 5. Comparing with scikit-learn
# ============================================================================
# - 使用 sklearn K-means
# - 对比性能
# - 绘制三方对比图

# ============================================================================
# 6. More
# ============================================================================
# 6.1 K-means++ initialization
# - 实现 K-means++ 并测试
# - 使用 make_blobs 生成 2 维数据

# 6.2 Generalize to any number of clusters
# - 测试不同簇数量（3、5、6）
# - 每个测试生成对比图
```

---

## 📦 提交说明

### 提交材料清单
1. ✅ **实验报告** (Word 或 PDF)
   - 使用提供的模板 `实验报告模板.md` 填写
   - 包含：实验目的、环境、过程、结果、分析、总结

2. ✅ **Python 代码文件**
   - 提交 `assignment_new.py`
   - 代码已按 PDF 顺序整合，包含标题注释

3. ✅ **图片输出** (可选)
   - `new_output/` 文件夹中的所有图片
   - 用于实验报告中的插图

### 打包命令

#### 方式 1: 打包完整文件夹

```powershell
# 在 machine_learning 目录下运行
Compress-Archive -Path task3 -DestinationPath task3_submission.zip -Force
```

#### 方式 2: 只打包必需文件

```powershell
# 在 task3 目录下运行
Compress-Archive -Path assignment_new.py,new_output,实验报告.docx,requirements.txt -DestinationPath ..\task3_submission.zip -Force
```

### 提交清单确认
- [ ] 实验报告（Word/PDF）
- [ ] assignment_new.py
- [ ] new_output/ 文件夹（包含所有图片）
- [ ] requirements.txt
- [ ] README.md（可选）

---

## ⚙️ 技术细节

### 算法实现亮点

1. **完整的 K-means 实现**
   - 支持随机初始化和 K-means++
   - 收敛条件判断
   - 空簇处理

2. **详细的中文注释**
   - 每个关键步骤都有说明
   - 算法原理解释
   - 参数说明

3. **性能评估**
   - Adjusted Rand Index
   - Silhouette Score
   - Inertia (簇内平方和)
   - 训练时间和迭代次数

4. **可视化**
   - 清晰的对比图
   - 质心标记
   - 性能指标显示

### 实验结果参考值

在 Iris 数据集上的典型结果：
- **ARI**: 约 0.45-0.80（取决于随机初始化）
- **Silhouette Score**: 约 0.52-0.55
- **sklearn ARI**: 约 0.79-0.80

---

## 🔍 常见问题

### Q1: 为什么我的 ARI 和报告中的不一样？
**A**: K-means 对初始值敏感，不同的随机种子会产生不同结果。这是正常现象。

### Q2: sklearn 警告 "memory leak on Windows with MKL"
**A**: 这是 sklearn 的已知警告，不影响结果。可通过设置环境变量 `OMP_NUM_THREADS=1` 来消除。

### Q3: 如何提高自定义 K-means 的性能？
**A**: 
1. 使用 K-means++ 初始化
2. 多次运行取最佳结果
3. 调整收敛阈值

### Q4: 图片编号的含义？
**A**: 
- `3_*.png` - 对应 PDF 第 3 章节
- `5_*.png` - 对应 PDF 第 5 章节
- `6_1_*.png` - 对应 PDF 第 6.1 小节
- `6_2_*.png` - 对应 PDF 第 6.2 小节

---

## 📚 参考资料

1. **课程讲义**: Lab03 - K-means Clustering
2. **scikit-learn 文档**: 
   - K-means: https://scikit-learn.org/stable/modules/clustering.html#k-means
   - Adjusted Rand Index: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
3. **原始论文**: Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding.

---

## 📅 时间安排

- **作业发布**: [填写]
- **截止日期**: [发布日期 + 1 周]
- **提交方式**: Canvas

---

## ✨ 特别说明

- ✅ 所有代码已测试通过
- ✅ 图片按章节编号
- ✅ 关键步骤有详细注释
- ✅ 支持任意簇数量
- ✅ 实现了 K-means++
- ✅ 与 sklearn 对比完整

---

**祝你作业顺利！如有问题，请查看代码注释或参考实验报告模板。** 🎓

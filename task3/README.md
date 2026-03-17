# K-Means Assignment

## 文件说明

- `kmeans_assignment.py`: 完整的作业脚本，**按照 lab03.pdf 章节顺序组织**
- `requirements.txt`: 依赖清单（numpy, matplotlib, scikit-learn）
- `outputs/`: 运行后生成的图片存放文件夹
- `lab03-1.pdf`: 作业说明文档

## 代码结构

所有代码已按照 PDF 顺序整合到 `kmeans_assignment.py` 中，每部分开始前有清晰的标题注释：

1. **# 1. K-means for the iris data** - 实现自定义 K-means 算法并应用于 Iris 数据集
2. **# 2. Plotting results** - 绘制聚类结果可视化图
3. **# 3. Measuring performance** - 计算并输出性能评估指标
4. **# 4. Comparing with scikit-learn** - 与 sklearn 实现对比
5. **# 5. More** - 使用 make_blobs 生成二维数据进行额外实验

## 运行方法

在 PowerShell 中运行：

```powershell
python kmeans_assignment.py
```

## 输出说明

运行后将：
- 在终端打印所有章节的执行结果和性能指标
- 在 `outputs/` 文件夹生成多张图片：
  - Iris 数据集的聚类结果对比图
  - make_blobs 数据集的算法结果与真实标签对比图（每个算法都有对比图）

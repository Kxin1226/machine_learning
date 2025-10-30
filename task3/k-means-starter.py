# k-means-starter.py
# parsons/28-feb-2017
#
# Running k-means on the iris dataset.
#
# Code draws from:
#
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()

#
# Put your K-means code here.
#

#
# Plot everything
#
plt.subplot( 1, 2, 1 )
# Plot the original data 
plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

plt.show()


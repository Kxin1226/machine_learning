#--
# classify-iris.py
# sklar/22-jan-2017
# This code demonstrates Decision Tree classification on the Iris data set, using the scikit-learn toolkit.
# Source is based on code from:
#  http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html
#--

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


#--
# define parameters
#--
n_classes = 3
plot_colors = "bry"
plot_step = 0.02


#--
# load data (Iris data set)
#--
iris = load_iris()
num_features = 4


#--
# classify and plot data
#--
plt.figure()
plt.rc( 'xtick', labelsize=8 )
plt.rc( 'ytick', labelsize=8 )
for i in range(0,num_features):
    for j in range(i+1,num_features):
        # classify using two corresponding features
        pair = [i, j]
        X = iris.data[:, pair]
        y = iris.target
        # train classifier
        clf = DecisionTreeClassifier().fit( X,  y )
        # plot the (learned) decision boundaries
        plt.subplot( num_features, num_features, j*num_features+i+1 )
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid( np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step) )
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.xlabel( iris.feature_names[pair[0]], fontsize=8 )
        plt.ylabel( iris.feature_names[pair[1]], fontsize=8 )
        plt.axis( "tight" )
        # plot the training points
        for ii, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == ii)
            plt.scatter(X[idx, 0], X[idx, 1], color=color, label=iris.target_names[ii])
        plt.axis("tight")
plt.show()

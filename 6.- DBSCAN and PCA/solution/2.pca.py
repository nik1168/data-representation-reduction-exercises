from __future__ import absolute_import, print_function, division
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from time import time
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset and remove mean

# load digits dataset
digits = load_digits()
X = digits.data
n_samples, n_features = X.shape
print("Data shape: ", X.shape)

x_bar = np.mean(X, axis=0)

# remove mean by subtracting the mean of variables
X_normalized = (X - x_bar)

# Step 2: Calculate covariance matrix

# In the lecture the covariance matrix is given by cov(x) = (1/n_samples)x*xt
# As we organize data matrix differently (rows are examples), we need to change the formula.
COV_X = np.cov(X_normalized.T)
C = ((1 / X_normalized.shape[0]) * np.matmul(X_normalized.T, X_normalized))

# visualize the covariance matrix, so that we can see the redundancy
plt.imshow(COV_X)

plt.show()
plt.imshow(C)
plt.show()



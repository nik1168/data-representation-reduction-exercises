import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error


## Step 1: Load data

def load_dat(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        dim = len(lines[0].strip().split())
        num_samples = len(lines)
        data = np.zeros((num_samples, dim))
        for i in range(num_samples):
            data[i, :] = np.array([float(x) for x in lines[i].strip().split()])
        return data


X = load_dat('ex1x.dat')
Y = load_dat('ex1y.dat')
# get some statistics of the data
num_samples = X.shape[0]  # get the first dimension of X (i.e. number of rows)
dim = X.shape[1]  # get the second dimension of X (i.e. number of columns)
print('X (%d x %d)' % (num_samples, dim))
print('Y (%d)' % (num_samples))

# Visualize the dataset in 3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=Y)
ax.set_xlabel('X[0]')
ax.set_ylabel('Y[1]')
ax.set_zlabel('Y')

plt.show()

## Step 2: Preprocess data
### add intercept term to all samples in X
intercept = np.ones([X.shape[0], 1])
X = np.hstack([X, intercept])
Y = Y.reshape([-1, 1])
print('X (%d x %d)' % (num_samples, dim + 1))
print('Y (%d x 1)' % (num_samples))


## Step 3: Fit the data

def pseudo_inverse(A):
    return np.linalg.pinv(A)


def sse(prediction, reference):
    size = reference.size
    return mean_squared_error(reference, prediction)*size


inverse = pseudo_inverse(np.matmul(np.transpose(X), X))
transY = np.matmul(np.transpose(X), Y)
beta = np.matmul(inverse, transY)
print("beta")
print(beta)

## Step 4: evaluate the model
# calculate the predicted scores
prediction = np.matmul(X,beta)  ### Your code here
print("Prediction")
print(prediction)
# calculate the sum of square error
error = sse(prediction, Y)
print('Sum of square error: %f' %error)

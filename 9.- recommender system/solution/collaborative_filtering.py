## Step 1: Load data and create masks

from __future__ import print_function, division
import numpy as np


# criteria RMSE
def RMSE(A, B, mask):
    rmse = np.sqrt(np.sum(mask * (A - B) ** 2) / np.sum(mask))
    return rmse


# criteria MAE
def MAE(A, B, mask):
    mae = np.sum(mask * np.abs(A - B)) / np.sum(mask)
    return mae


def load_rating(fname, N=943, M=1682):
    ''' load rating file with the format: UserID::MovieID::Rating::Timestamp
    Can be used with MovieLens100K & MovieLens1M
    Params:
        - fname: file name
        - N: number of users
        - M: number of items (e.g. movies)
    '''
    R = np.zeros((N, M))
    with open(fname, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            splt = line.strip().split('\t')
            uid = int(splt[0]) - 1
            mid = int(splt[1]) - 1
            r = float(splt[2])
            R[uid, mid] = r
    return R


# load training and testing sets
data_train = load_rating("u1.base").T
data_test = load_rating("u1.test").T
n_movies_train, n_users_train = data_train.shape
n_movies_test, n_users_test = data_test.shape
print("Finish")


# create mask matrix
# X: The rating matrix with size (n_movies,n_users)
# Return: Binary mask matrix where 1 indicates there is a rating and 0 vice versa
def create_mask(X):
    B = X > 0
    return B.astype(np.int)


## Step 2: Implement functions to calculate cost and gradients
# This function computes the cost value that we want to minimize
# THETA: A matrix contains users' feature
# X: A matrix contains movies' feature
# Y: A matrix contains ground truth (n_movies x n_users)
# _lambda: Regularization parameter
# mask: The binary mask matrix
def compute_cost(X, THETA, Y, _lambda, mask):
    assert X.shape[1] == THETA.shape[1]
    assert X.shape[0] == Y.shape[0]
    assert THETA.shape[0] == Y.shape[1]
    assert Y.shape == mask.shape
    n_movies_train, n_users_train = X.shape
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(n_movies_train):
        for j in range(n_users_train):
            if mask[i][j] == 1:
                sum1 = sum1 + (np.matmul(THETA[j].T, X[i]) - Y[i][j]) ** 2
    n_m = n_movies_train
    n_u = n_users_train
    for i in range(n_m):
        for k in range(n_features):
            sum2 = sum2 + (X[i][k]) ** 2

    for i in range(n_u):
        for k in range(n_features):
            sum3 = sum3 + (X[i][k]) ** 2

    return ((1 / 2) * sum1) + ((_lambda / 2) * sum2) + ((_lambda / 2) * sum3)


# This function computes partial derivatives of the cost function with regards to movie and user features
# THETA: A matrix contains users' feature
# X: A matrix contains movies' feature
# Y: A matrix contains ground truth (n_movies x n_users)
# _lambda: Regularization parameter
# mask: The binary mask matrix
# return: a tuple (grad_X,grad_THETA)
def compute_gradient(X, THETA, Y, _lambda, mask):
    assert X.shape[1] == THETA.shape[1]
    assert X.shape[0] == Y.shape[0]
    assert THETA.shape[0] == Y.shape[1]
    assert Y.shape == mask.shape
    grad_X = X - np.dot(alpha, np.dot(np.dot((np.dot(THETA.T, X) - Y), THETA), mask) + np.dot(_lambda, X))
    grad_THETA = THETA - np.dot(alpha,
                                np.dot(np.dot((np.dot(THETA.T, X) - Y), X), mask) + np.dot(_lambda, THETA))
    return (grad_X, grad_THETA)


## Step 3: Training

# %%

n_features = 10
MOVIE_FEATURES = 0.25 * np.random.randn(n_movies_train, n_features)
USER_FEATURES = 0.25 * np.random.randn(n_users_train, n_features)
_lambda = 0.01
mask = create_mask(data_train)
alpha = 0.001
training_epochs = 150
counter = 0

while counter < training_epochs:
    # Compute gradients
    grad_X, grad_THETA = compute_gradient(MOVIE_FEATURES, USER_FEATURES, data_train, _lambda, mask)

    # update parameters here
    MOVIE_FEATURES = MOVIE_FEATURES - alpha * grad_X
    USER_FEATURES = USER_FEATURES - alpha * grad_THETA

    # compute cost function
    cost = compute_cost(MOVIE_FEATURES, USER_FEATURES, data_train, _lambda, mask)

    # increase counter
    counter += 1
    print("epoch:", counter, "cost: ", cost)

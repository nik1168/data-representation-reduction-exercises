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


# create mask matrix
# X: The rating matrix with size (n_movies,n_users)
# Return: Binary mask matrix where 1 indicates there is a rating and 0 vice versa
def create_mask(X):
# YOUR CODE GOES HERE

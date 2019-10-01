#!/usr/bin/env python
# coding: utf-8

from sklearn import datasets
import numpy as np


# load dataset
X, y = datasets.load_boston(return_X_y=True)
print(X.shape)
print(y.shape)


# create virtual features, including
#   second degree of the first variable
#   second degrees of the eighth variable
#   third and second degrees of the eleventh variable
### Your code here ###

# concatenate the virtual feature to the original features
### Your code here ###

# add a dimension with all 1 to account for the intercept term
interc = np.ones((X.shape[0], 1))
X = np.hstack((interc, X))
print(X.shape)


# split training and testing dataset
train_ratio = 0.8
cutoff = int(X.shape[0] * train_ratio)
X_tr = X[:cutoff, :]
y_tr = y[:cutoff]
X_te = X[cutoff:,:]
y_te = y[cutoff:]
print('Train/Test: %d/%d' %(X_tr.shape[0], X_te.shape[0]))


# linear regression using the normal equation
def pseudo_inverse(A):
    # Calculate the pseudo_inverse of A
    pinv = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
    return pinv 


# fit the polynomial on the training set
beta = ### Your code here ###


# evaluation functions
def MSE(prediction,reference):
    # Calculate the mean square error between the prediction and reference vectors
    mse = ### Your code here ###
    return mse 

def MAE(prediction, reference):
    # Calculate the mean absolute error between the prediction and reference vectors
    mae = ### Your code here ###
    return mae 


# make prediction on the testing set
pred = ### Your code here ###
mse = MSE(pred, y_te)
mae = MAE(pred, y_te)
print(mse)
print(mae)


# regularized linear regression 
def regularized_pseudo_inverse(A, theta):
    # Calculate the regularized pseudo_inverse of A
    ### Your code here ###
    return pinv 


# fit the polynomial, regularized by theta
theta = 0.5
beta_regularized = ### Your code here ###  


# make prediction on the testing set
pred_2 = ### Your code here ###
mse = MSE(pred_2, y_te)
mae = MAE(pred_2, y_te)
print(mse)
print(mae)


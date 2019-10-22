# Step 1: Load dataset, split into training and test sets, and scale features
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# load boston housing price dataset
from sklearn.utils import shuffle

boston = load_boston()
x = boston.data
y = boston.target

# split into training and test sets, namely 80 percent of examples goes for the training, 20 percent goes for the test set
N_train = int(0.8 * x.shape[0])
x_train = x[:N_train, :]
y_train = y[:N_train]
x_test = x[N_train:, :]
y_test = y[N_train:]

# scale features by removing mean and dividing by the standard deviation
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

print(x_train_scaled.shape)
print(y_train.shape)
print(x_test_scaled.shape)
print(y_test.shape)

# Step 2: Add intercept terms and initialize parameters
# Note: If you run this step again, please run from step 1 because notebook keeps the value from the previous run
interc_train = np.ones((x_train_scaled.shape[0], 1))
interc_test = np.ones((x_test_scaled.shape[0], 1))
x_train_scaled = np.hstack((interc_train, x_train_scaled))

x_test_scaled = np.hstack((interc_test, x_test_scaled))

print(x_train_scaled.shape)
print(x_test_scaled.shape)

# init parameters using random values
number_of_features = x_train_scaled.shape[1]
mu, sigma = 0, 0.5
theta = np.random.normal(mu, sigma, number_of_features)
print(theta)


# Step 3: Implement the gradient and the cost function
# In this step, you have to calculate the gradient. You can use the provided formula but the best way is to vectorize
# that formula for efficiency
def compute_gradient(x, y, theta):
    n = x.shape[0]
    return np.matmul(x.T, (np.matmul(x, theta) - y)) / n


def compute_cost(x, y, theta):
    x_theta = np.matmul(x, theta)
    x_theta_minus_y = x_theta - y
    n = x.shape[0]
    return (1 / (2 * n)) * np.matmul(x_theta_minus_y.T, x_theta_minus_y)


# Step 4: stochastic gradient descent
import matplotlib.pyplot as plt
import copy

# try different values for the learning rate
learning_rate = 0.01

# number of training iterations
num_samples = x_train_scaled.shape[0]
N_iterations = num_samples * 20  # loop over the training dataset 20 times

# prepare to plot
plt.subplot(111)

# calculate cost value and update theta
J = np.zeros(N_iterations)

# initialize new parameters using random distribution
theta_sgd = 0.5 * np.random.randn(x_train_scaled.shape[1])
idx = 0
c = 0
for step in range(N_iterations):
    if step % num_samples == 0:
        # shuffle the training data (must be done the same way for data and targets)
        # YOUR CODE GOES HERE
        x_train_scaled, y_train = shuffle(x_train_scaled, y_train, random_state=0)
        c = 0
    idx = step - int(step / num_samples) * num_samples
    # select the next sample to train
    x_step = x_train_scaled[idx]
    y_step = y_train[idx]
    x_step = x_step.reshape([1, -1])
    c += 1

    # calculate the cost on x_step and y_step
    J[step] = compute_cost(x_step, y_step, theta_sgd)

    # update theta using a x_step and y_step
    grad = compute_gradient(x_step, y_step, theta_sgd)
    theta_sgd = theta_sgd - (learning_rate * grad)

# calculate the loss on the whole training set
J_train = compute_cost(x_train_scaled, y_train, theta_sgd)
print('training cost: %f' % J_train)
# plot cost function
plt.plot(J)
plt.xlabel('Training step')
plt.ylabel('Cost')
plt.show()

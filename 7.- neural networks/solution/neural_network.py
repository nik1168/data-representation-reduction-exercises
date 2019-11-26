## Step 1: Dataset loading

from __future__ import absolute_import, division, print_function
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score

# load digits dataset with 5 classes. The dataset has 10 classes in total.
# You can change the amount of data as you like.

num_classes = 5
digits = load_digits(n_class=num_classes)
x = digits.data
y = digits.target
n_samples, n_features = x.shape

print("data shape: ", x.shape)
print("class shape: ", y.shape)

N_train = int(0.8 * x.shape[0])
x_train = x[:N_train, :]
y_train = y[:N_train]
x_test = x[N_train:, :]
y_test = y[N_train:]

# Add the bias term
intercept_train = np.ones((N_train, 1))
x_train = np.hstack((intercept_train, x_train))

intercept_test = np.ones((x.shape[0] - N_train, 1))
x_test = np.hstack((intercept_test, x_test))

# Convert labels to one-hot vector
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train_onehot = onehot_encoder.fit_transform(integer_encoded)
print(y_train_onehot)
print("Finish onehot encode")

"""
When using a sigmoid activation with one-hot labels for classification, the network outputs a probability 
for each possible class. This is a clear advantage over using the original form of labels. For example, when 
the network predicts a sample as number 1 and number 3 with 50% and 40% probabilities, respectively, we know 
that the sample could be a number 3, but it will be more likely to be a number 1. If we don't use one-hot encoding, 
the output would then likely be in the range of number 2, which would be completely wrong.
Check whether your one-hot conversion above is correct or not by the following
"""
print(y_train[:5])
print(y_train_onehot[:5])

## Step 2: Forward computation
"""
Most deep learning frameworks provide a list of basic functions as building blocks, such as fully_connected, sigmoid, relu... 
so that you can stack them sequentially as layers to build your own neural networks. In this exercise, we will see implement 
the fully connected layer and the sigmoid activation function.
 In each function, we will return the result and the cache the input for backward computation later
"""


def sigmoid(x):
    cache = x
    result = 1.0 / (1 + np.exp(-x))
    return cache, result


def fully_connected(x, theta):
    cache = (x, theta)
    result = np.matmul(x, theta)
    return cache, result


# After having our building blocks, we can start stacking layers.
def compute_forward(x, theta_matrices):
    '''
    x: feature vector
    theta_matrices: The list contains all theta. The first element is the theta (matrix) of the input layer and the first hidden
    layer, the second one is the theta of the fist hidden layer and the second hidden layer, and so on

    In this exercise, our network architecture will be:
    input -> fully_connected -> sigmoid -> fully_connected -> sigmoid -> output
    You don't need to use regularization in this exercise
    '''
    result = x
    cache = dict()
    for i, theta in enumerate(theta_matrices):
        cache_layer, fully_connected_layer = fully_connected(result, theta)
        cache_sigmoid_layer, sigmoid_to_layer = sigmoid(fully_connected_layer)
        cache['fc' + str(i) + ''] = cache_layer
        cache['sigmoid' + str(i) + ''] = sigmoid_to_layer
        result = sigmoid_to_layer
    ## Your code here, should be a result of a fully_connected layer then a sigmoid activation.
    # Store the result of each computation in cache, for doing backprop later.
    # For this exercise, cache should have four items with keys: fc0, sigmoid0, fc1, sigmoid1
    return cache, result


def compute_cost(outputs, labels):
    '''mean square error'''
    result = mean_squared_error(labels, outputs)
    return result


# %%

num_hidden = 100
theta0 = np.random.normal(loc=0., scale=0.5, size=(n_features + 1, num_hidden + 1))  # + 1 for bias term
theta1 = np.random.normal(loc=0., scale=0.5, size=(num_hidden + 1, num_classes))
theta_matrices = [theta0, theta1]
cache, initial_outputs = compute_forward(x_train, theta_matrices)
assert initial_outputs.shape == y_train_onehot.shape, 'forward pass returns wrong shape'
print('forward pass returns correct shape')
initial_cost = compute_cost(initial_outputs, y_train_onehot)
print("Initial cost")
print(initial_cost)

## Step 3: Backpropagation
"""
Similar to forward pass, calculating backward gradient using backpropagation 
is just like stacking several layers of gradient together. 
To do so, we first need to calculate the gradient of each of our building blocks.
"""


def fc_backward(cache, result):
    x, theta = cache
    theta_grad = x.T.dot(result)
    x_grad = result.dot(theta.T)
    return x_grad, theta_grad


def sigmoid_backward(cache, result):
    x = cache
    sigmoid_grad = sigmoid(x)[1] * (1 - sigmoid(x)[1])
    return sigmoid_grad * result


def cost_backward(outputs, labels):
    # Derivative of block of cost_backward function
    result = -(labels - outputs)
    return result


def compute_backprop(x, theta_matrices, cache, outputs, labels):
    '''
    return gradients for theta_matrices
    '''
    theta_grad = {
        'theta0': None,
        'theta1': None
    }  # should include two key theta0 and theta1 for this exercise
    grad = cost_backward(outputs, labels)
    for i, theta in enumerate(theta_matrices[::-1]):
        layer = len(theta_matrices) - i - 1  # first iteration: layer 1, second iter: layer 0
        # Your code, first you need to propagate the gradient through the sigmoid activation,
        # then through the fully_connected layer
        res_sig_back = sigmoid_backward(cache['sigmoid' + str(layer)], grad)
        xgrad, res_fc_backward = fc_backward(cache['fc' + str(layer)], res_sig_back)
        grad = xgrad
        theta_grad['theta' + str(layer)] = res_fc_backward

    return theta_grad


#### Check if compute_backprop returns the right shape
theta_grad = compute_backprop(x_train, theta_matrices, cache, initial_outputs, y_train_onehot)
dtheta0 = theta_grad['theta0']
assert dtheta0.shape == theta0.shape, 'backprop returns wrong shape for theta 0'
dtheta1 = theta_grad['theta1']
assert dtheta1.shape == theta1.shape, 'backprop returns wrong shape for theta 1'

# %% md

## Step 4: Training the network

# %% md

# Now that you have both forward and backward computation, use batch gradient descent to train the network.
alpha = 0.001  # learning rate
N_iterations = 200  # You can play with this one to see how the network performs
J = np.zeros(N_iterations)

for i in range(N_iterations):
    ## Your code
    # First, do a forward pass
    cache, initial_outputs = compute_forward(x_train, theta_matrices)

    # Calculate the cost and store it in J[i]
    cost = compute_cost(initial_outputs, y_train_onehot)
    print("cost it: " + str(i))
    print(cost)
    J[i] = cost

    # Calculate the gradients by doing a backward pass
    gradients = compute_backprop(x_train, theta_matrices, cache, initial_outputs, y_train_onehot)

    # Update the weights by gradient descent rule
    theta_matrices[0] = theta_matrices[0] - alpha * gradients['theta0']
    theta_matrices[1] = theta_matrices[1] - alpha * gradients['theta1']

J_train = compute_cost(compute_forward(x_train, theta_matrices)[1], y_train_onehot)
print('training cost: %f' % J_train)

# plot cost function
plt.plot(J)
plt.xlabel('Training step')
plt.ylabel('Cost')
plt.show()


## Step 5: Evaluation
# Your accuracy on test set should be greater than 90%

def compute_accuracy(y_ground_truth, y_pred):
    return accuracy_score(y_ground_truth, y_pred)


pred_one_max = compute_forward(x_test, theta_matrices)[1]
print("Prediction")

pred = np.array(list(map(lambda pred: np.argmax(pred), pred_one_max)))
# Your prediction would be an one-hot vector, for each test sample, select the one with the highest probablity to assign the class
accuracy = compute_accuracy(y_test, pred)
print('accuracy: ', accuracy)

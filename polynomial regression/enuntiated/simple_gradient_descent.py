
# Step 1: Load dataset, split into training and test sets, and scale features
import numpy as np
from sklearn.datasets import load_boston

# load boston housing price dataset
boston = load_boston()
x = boston.data
y = boston.target

# split into training and test sets, namely 80 percent of examples goes for the training, 20 percent goes for the test set
N_train = int(0.8 * x.shape[0])
x_train = x[:N_train,:]
y_train = y[:N_train]
x_test = x[N_train:,:]
y_test = y[N_train:]

# scale features by removing mean and dividing by the standard deviation
x_train_scaled = # YOUR CODE GOES HERE
x_test_scaled = # YOUR CODE GOES HERE

print(x_train_scaled.shape)
print(y_train.shape)
print(x_test_scaled.shape)
print(y_test.shape)


# Step 2: Add intercept terms and initialize parameters
# Note: If you run this step again, please run from step 1 because notebook keeps the value from the previous run
x_train_scaled = # YOUR CODE GOES HERE

x_test_scaled = # YOUR CODE GOES HERE

print(x_train_scaled.shape)
print(x_test_scaled.shape)

# init parameters using random values
theta = # YOUR CODE GOES HERE
print(theta)


# Step 3: Implement the gradient and the cost function
# In this step, you have to calculate the gradient. You can use the provided formula but the best way is to vectorize
# that formula for efficiency
def compute_gradient(x,y,theta):
    # YOUR CODE GOES HERE

def compute_cost(x,y,theta):
    # YOUR CODE GOES HERE


# Step 4: Verify the gradient value
# In this step, you need to verify that the computed gradient is correct. The difference betweet the gradient and the
# approximate gradient should be very small (~10^-18)
def approximate_gradient(x,y,theta,epsilon):
    n_features = x.shape[1]
    app_grad = np.zeros(n_features)
    for i in range(n_features):
        app_grad[i] = # YOUR CODE GOES HERE
    return app_grad

grad = compute_gradient(x_train_scaled,y_train,theta)
epsilon = 1e-4
app_grad = approximate_gradient(x_train_scaled,y_train,theta,epsilon)
print('Sum of gradient squared error: ',np.sum((grad - app_grad)**2))


# Step 5: Try gradient descent algorithm with different learning rates
import matplotlib.pyplot as plt
import copy

# try different values for the learning rate
learning_rates = [0.001,0.003,0.01,0.03,0.1,0.3]

# this matrix keeps the learned parameters
theta_matrix = np.zeros((len(learning_rates),x_train_scaled.shape[1]))

# number of training iterations
N_iterations = 100

# prepare to plot
plt.subplot(111)

# calculate cost value and update theta
for indx,alpha in enumerate(learning_rates):
    # keep the cost value for each training step
    J = np.zeros(N_iterations)
    
    # initialize new parameters using random distribution
    theta = 0.5 * np.random.randn(x_train_scaled.shape[1])
    for step in range(N_iterations):
        # update theta
        theta = # YOUR CODE GOES HERE
        
        # save the value of theta
        theta_matrix[indx,:] = theta
        
        # calculate the cost on traing set
        J[step] = # YOUR CODE GOES HERE
    # plot cost function
    plt.plot(J)
plt.xlabel('Training step')
plt.ylabel('Cost')
plt.legend(('0.001','0.003','0.01','0.03','0.1','0.3'), loc='upper right')
plt.show()
        

# Step 6: Predict the price of house
# You have to select the best theta you found
theta = # YOUR CODE GOES HERE
predict_price = # YOUR CODE GOES HERE

# calculate the cost for the test set
test_cost = # YOUR CODE GOES HERE
print('test cost: ',test_cost)

# plot the ground truth and the predicted
x_axis = np.linspace(1,len(y_test),len(y_test))
plt.plot(x_axis,y_test,'b',x_axis,predict_price,'r')
plt.legend(('Ground truth','Predicted'))
plt.show()





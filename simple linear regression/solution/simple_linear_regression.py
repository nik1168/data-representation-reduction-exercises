import numpy as np
import matplotlib.pyplot as plt

# generate an array of floats between 0 and 1
number_of_features = 10
x = np.random.uniform(low=0, high=1, size=(number_of_features,))  # Write your code here
y = 1 + 2 * x
# plot data
# plt.plot(x,y,'b')
# plt.show()

# # Step 2: Generate noisy data
# generate random values using normal distribution with mean = 0 and standard deviation = 0.1
mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, number_of_features)  # Write your code here
y_noise = y + noise
#
plt.plot(x, y, 'b', x, y_noise, 'rx')
plt.show()

#
## Step 3: Find parameters
# Here you have to use the provided formulas to calculate the parameters theta_0 ($\theta_0$) and theta_1 ($\theta_1$)
def get_theta_1(x1, y):
    X = np.stack((x1, y), axis=0)
    cov = np.cov(X)[0][1]
    var = np.var(x1)
    return cov / var


def get_theta_2(x1, y, theta_1):
    return np.mean(y) - theta_1 * np.mean(x1)


# Write your code below
theta_1 = get_theta_1(x, y_noise)
theta_0 = get_theta_2(x, y_noise, theta_1)
#
z = theta_0 + theta_1 * x
#
# plot the original line and the fitting linear line here on the same graph
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y,'b',x,y_noise,'rx',x,z,'g')
plt.legend(('Original Data', 'Noisy Data', 'Linear Fit'), loc='lower right')
plt.show()
#
## Step 4: Fit multiple lines
plt.subplot(111)
num_sigma = 6
y_noise = np.zeros((num_sigma,len(x)))
z = np.zeros((num_sigma,len(x)))
for indx,sigma in enumerate(np.linspace(0,1,num=num_sigma)):
    # generate the noise
    noise = np.random.normal(mu, sigma, number_of_features)
    y_noise[indx,:] = noise + y
    # calculate theta_1 and theta_0
    theta_1 = get_theta_1(x, y_noise[indx,:])
    theta_0 = get_theta_2(x,y_noise[indx,:],theta_1)
    z[indx,:] = theta_0 + theta_1*x
    # plot the fitting line
    plt.plot(x,z[indx,:])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(('0.0','0.2','0.4','0.6','0.8','1.0'), loc='lower right')
#
# reset the color cycle, so that the same color will be used for dots
plt.gca().set_prop_cycle(None)
# plot dots in the same figure
for indx in range(num_sigma):
    plt.plot(x,y_noise[indx,:],'.')
plt.show()
#
## Step 5: Explore the statistical properties of simple linear regression
mu = 0
sigma = 0.1
n = 1000
params = np.zeros((n,2))
for i in range(n):
    # genrate the noise
    noise = np.random.normal(mu, sigma, number_of_features)
    y_noise = y + noise
    # your code goes here
    theta_1 = get_theta_1(x, y_noise)
    theta_0 = get_theta_2(x,y_noise,theta_1)
    params[i,0] = theta_0
    params[i,1] = theta_1
theta_bar = np.mean(params,axis=0)
theta_sd = np.std(params,axis=0)
print("Mean Theta 0: ",theta_bar[0])
print("Mean Theta 1: ",theta_bar[1])
print("STD 0: ",theta_sd[0])
print("STD 1: ",theta_sd[1])
f, axarr = plt.subplots(2, sharex=True)
axarr[0].hist(params[:,0],bins=30)
axarr[0].set_title('Theta 0')
axarr[1].hist(params[:,1],bins=30)
axarr[1].set_title('Theta 1')
plt.show()

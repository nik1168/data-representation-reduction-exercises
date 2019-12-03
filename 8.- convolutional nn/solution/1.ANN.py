# Run the below to set up the notebook, you need to have Tensorflow installed for this exercise.

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

## Step 1: Load dataset

num_classes = 10
digits = load_digits(n_class=num_classes)
x = digits.data
y = digits.target
n_samples, n_features = x.shape
shape_image = x.shape[1]

# plt.imshow(x[0])
print("data shape: ", x.shape)
print("class shape: ", y.shape)

# Split the data into training and testing sets
N_train = int(0.8 * x.shape[0])
x_train = x[:N_train, :]
y_train = y[:N_train]
x_test = x[N_train:, :]
y_test = y[N_train:]

# Convert labels to one-hot vector
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train_onehot = onehot_encoder.fit_transform(integer_encoded)
print(y_train_onehot)
print("Finish onehot encode")

## Step 2: Build a neural network model to classify the digits
# One of the most simplest ways to build a neural network with Tensorflow is to use high-level interfaces from Keras

# First, define the based sequential model
model = Sequential()
# Add the first fully connected layer with 100 hidden units, with ReLU activation.
# As this is the first layer in your model, don't forget to include the 'input_shape' argument
model.add(Dense(100, activation='relu', input_shape=(shape_image,)))
model.add(Dense(10, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'],
)

## Step 3: Train the model
model.fit(x_train, y_train_onehot, epochs=10, batch_size=32)

## Step 4: Evaluate the model
# Call 'predict' function from the model to get the predicted class probabilities
y_predict = model.predict(x_test)
# Find the prediction (as the classes with highest probabilities)
y_predict_max = np.array(list(map(lambda row: np.argmax(row), y_predict)))

# Calculate the prediction accuracy
accuracy = accuracy_score(y_test, y_predict_max)
print("Accuracy={:.2f}".format(accuracy))

## Step 5: Visualize the classification results
for selected_class in range(0,10):
    x_visualize = x_test[y_predict_max == selected_class]
    # plot some images of the digits
    n_img_per_row = 10
    img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 10 * i + 1
        for j in range(n_img_per_row):
            iy = 10 * j + 1
            if i * n_img_per_row + j < len(x_visualize):
                img[ix:ix + 8, iy:iy + 8] = x_visualize[i * n_img_per_row + j].reshape((8, 8))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('Test images predicted as "{:}"'.format(selected_class))
    plt.show()

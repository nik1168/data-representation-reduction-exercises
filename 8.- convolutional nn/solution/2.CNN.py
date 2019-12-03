from __future__ import absolute_import, division, print_function
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
import numpy as np
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt

## Step 1: Load dataset

num_classes = 10
digits = load_digits(n_class=num_classes)
x = digits.data
y = digits.target

H = 8
W = 8
C = 1
# reshape x into a numpy array of images of size: height x width x number_of_channels
x = x.reshape((-1, H, W, C))

print("data shape: ", x.shape)
print("class shape: ", y.shape)

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

## Step 2: Build a CNN model to classify digits

# First, define the based sequential model
model = Sequential()

# Add the first convolutional layer:
# A 2D Convolution layer with:
#     32 filters
#     kernel size: 3x3
#     stride: 1
#     padding scheme: 'same'
#     use_bias: True
#     activation: relu
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(H, W, 1),
                 strides=1,
                 padding='same',
                 use_bias=True))

# Add the second convolutional layer:
# A 2D Convolution layer with:
#     64 filters
#     kernel size: 3x3
#     stride: 1
#     padding scheme: 'same'
#     use_bias: True
#     activation: relu

model.add(Conv2D(64,
                 kernel_size=(3, 3),
                 activation='relu',
                 strides=1,
                 padding='same',
                 use_bias=True))

# Add dropout layer with rate = 0.75
model.add(Dropout(0.75))

# Add 2D max pooling layer with pooling size = 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add Flatten layer to reshape the output of the previous layer into 1D vectors
model.add(Flatten())

# Fully connected layers to perform the classification from the outputs of the previous layers
# Add a fully connected layer with 10 output units, and softmax activation
# (each hidden unit corresponds to one class, i.e. digit):
model.add(Dense(10, activation='softmax'))

# Compile the model with 'categorical_crossentropy' loss function and
# 'sgd' optimizer
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

## Step 3: Train the model
model.fit(x_train, y_train_onehot, epochs=10, batch_size=32)
scores = model.predict(x_test)
y_pred = np.argmax(scores, axis=1)
print("Accuracy={:.2f}".format(np.mean(y_pred == y_test)))

# %% md

## Step 5: Visualize the classification results

# %%

for selected_class in range(0, 10):
    x_visualize = x_test[y_pred == selected_class]
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

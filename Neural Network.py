import numpy as np

"""
Author: Rambod Azimi
This python code implements a 2-layer neural network
without using any frameworks (i.e. TensorFlow, PyTorch, ...)

Before running the code, make sure that you have already installed all the required libraries in this code to your computer
"""

# Defining a sample training dataset consisting of 20 elements with 2 feature variables
X_train = np.array([[185.32,  12.69],[259.92,  11.87],[231.01,  14.41], [175.37,  11.72],[187.12,  14.13],[225.91,  12.1 ],[208.41,  14.18],
                    [207.08,  14.03],[280.6 ,  14.23],[202.87,  12.25],[196.7 ,  13.54], [270.31,  14.6 ],[192.95,  15.2 ],[213.57,  14.28],
                    [164.47,  11.92],[177.26,  15.04],[241.77,  14.9 ],[237.  ,  13.13],[219.74,  13.87], [220.0, 13.9]])

Y_train = np.array([[0.],[0.],[0.],[1.],[1.],[0.],[0.],[0.],[1.],[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])
      
# Printing the size of the 2D array
print(X_train.shape)
print(Y_train.shape)

# Let's normalize our data for better accuracy, using TensorFlow
import tensorflow as tf
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(X_train)
normalized_X_train = norm(X_train)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense_function(input, w, b):
    units = w.shape[1] # get the number of neurons in the layer
    output = np.zeros(units)

    for i in range(units):
        z = np.dot(w[:,i], input) + b[i]
        output[i] = sigmoid(z)
    
    return output

# Building a 2-layer neural network
def sequential_function(x, w1, b1, w2, b2):
    a1 = dense_function(x, w1, b1)
    a2 = dense_function(a1, w2, b2)
    return a2

# Now, let's put the parameter values computed before
w1_tmp = np.array([[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]])
b1_tmp = np.array([-9.82, -9.28,  0.96])
w2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
b2_tmp = np.array([15.41])

# Make the prediction by running the built model
def predict(x, w1, b1, w2, b2):
    m = x.shape[0] # m = number of examples (20)
    p = np.zeros((m, 1))

    for i in range(m):
        p[i, 0] = sequential_function(x[i], w1, b1, w2, b2)

    return p

# Now, let's test the model
X_test = np.array([[200.0, 13.9],
                   [200.0, 17.0]])
normalized_X_test = norm(X_test)
predictions = predict(normalized_X_test, w1_tmp, b1_tmp, w2_tmp, b2_tmp)

y = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        y[i] = 1
    else:
        y[i] = 0
print(f"decisions = \n{y}")

# 2-Layer Neural Network for Binary Classification

## Introduction
This Python code implements a simple 2-layer neural network for binary classification. The neural network is trained to classify input samples into one of two classes (0 or 1) based on their features. The implementation does not rely on any deep learning frameworks like TensorFlow or PyTorch, except for data normalization, which utilizes TensorFlow.

## Author
Author: Rambod Azimi

## Prerequisites
Before running the code, make sure you have the following libraries installed on your computer:
NumPy: For numerical computations.
TensorFlow: For data normalization only.

## Training Data
The sample training dataset consists of 20 elements, with each element having 2 feature variables (X_train). Additionally, corresponding labels for each example (Y_train) are provided. The goal is to train the neural network to classify the elements based on these features into two classes: 0 or 1.

## Normalization
Before training the neural network, the training data (X_train) is normalized for better accuracy. Data normalization helps bring all features to a similar scale and avoid numerical instability during training. TensorFlow's Normalization layer is used for this purpose.

## Activation Function
The code includes the sigmoid activation function, which is commonly used in binary classification tasks. The sigmoid function maps any real-valued number to a value between 0 and 1.

## Neural Network Architecture
The neural network consists of two layers:

Input layer: The first layer receives the input features and applies a sigmoid activation function.
Output layer: The second layer takes the output of the first layer as input and produces the final prediction. Again, a sigmoid activation function is applied to the output.

## Helper Functions
dense_function(input, w, b): This function calculates the output of a dense layer given the input, weights (w), and biases (b). It iterates over each neuron in the layer and applies the sigmoid activation function to produce the output.

sequential_function(x, w1, b1, w2, b2): This function builds the 2-layer neural network by stacking two dense layers on top of each other. It first applies dense_function with the weights and biases of the first layer (w1 and b1) to the input x, and then applies the second dense layer using the weights and biases of the second layer (w2 and b2) to the output of the first layer.

predict(x, w1, b1, w2, b2): This function makes predictions using the trained neural network model. It takes an input array x containing test examples and the trained weights and biases of the two layers (w1, b1, w2, b2). It iterates through each test example, computes the output of the neural network using sequential_function, and stores the predictions in a separate array.

Pre-Trained Model Parameters
The code assumes that the parameters w1_tmp, b1_tmp, w2_tmp, and b2_tmp are pre-trained or have been obtained from a previously trained model. These parameters are used for testing the model on new data.

## Testing the Model
The model is tested on a separate test dataset (X_test). Before making predictions, the test data is normalized using the same normalization parameters obtained during training. The predictions are then binary decisions, where values greater than or equal to 0.5 are classified as class 1, and values less than 0.5 are classified as class 0.

## Output
The binary decisions for the test dataset are printed as a 2D array (y), where each element corresponds to the predicted class (0 or 1) for each test example.

## Important Note
To achieve satisfactory accuracy and meaningful predictions, it is crucial to ensure that the provided pre-trained parameters (w1_tmp, b1_tmp, w2_tmp, and b2_tmp) are appropriate for the specific dataset being used. If the parameters are not optimized or trained on the same dataset, the predictions may not be accurate.

## Usage
To use this code for your own binary classification task, follow these steps:

Define your training dataset with features (X_train) and corresponding labels (Y_train).
Normalize the training data using TensorFlow's normalization layer (as shown in the code).
Train the neural network model on your dataset to obtain the optimized weights and biases for each layer.
Save the trained parameters (w1, b1, w2, b2) for future use.
For testing, define a separate test dataset (X_test) and normalize it using the same normalization parameters from the training step.
Use the predict function with the trained parameters to make predictions on the test data.

## Conclusion
This code demonstrates a simple implementation of a 2-layer neural network for binary classification using only NumPy, without relying on deep learning frameworks. By understanding and adapting the code, you can apply similar techniques to other classification tasks and expand it to more complex neural network architectures.

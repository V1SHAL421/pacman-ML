# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np

class Classifier: # Use numpy and Multiclass Logistic Regression
    # Based off data, classifier should be able to choose which target (action) from 0-3
    def __init__(self):
        self.n_iterations = 150 # Number of iterations
        self.learning_rate = 0.01 # Learning rate of model
        self.weights = np.zeros((25, 4)) # Initialise weights vector to 0
        self.biases = np.zeros(4) # Initialise bias vector to 0
    
    def reset(self):
        self.weights = np.zeros_like(self.weights) # Reinitialise weights to 0
        self.biases = np.zeros_like(self.biases) # Reinitialise biases to 0
    
    def fit(self, data, target):
        # Convert input arrays into numpy arrays
        np_data = np.asarray(data)
        np_target = np.asarray(target)
        
        np_one_hot_encode_target = np.eye(4)[np_target]
        # Ensure feature matrix and target vector are aligned
        print(f"Shape of the feature matrix is {np_data.shape} and the shape of the target vector is {np_target.shape}")

        counter = 0
        for iteration in range(self.n_iterations):

            softmax_input = np.dot(np_data, self.weights) + self.biases


            # Softmax function


            softmax_input = np.atleast_2d(softmax_input) # Ensures the softmax input is at least 2-dimensional
            exp_logits = np.exp(softmax_input - np.max(softmax_input, axis=1, keepdims=True)) # Convert input into stabilized and exponentiated logits
            softmax_vector = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # Produces probabilities


            # Cross Entropy Loss

            
            softmax_vector = np.clip(softmax_vector, (1e-12), 1. - (1e-12)) # Ensures all values are in the range [1e-12, 1 - 1e-12]
            loss = (np.sum(np_one_hot_encode_target * np.log(softmax_vector)) / np_one_hot_encode_target.shape[0])* (-1) # Compute loss

            if iteration % 5 == 0:
                print(f"Iteration {iteration}, Loss: {loss}") # Print iteration and loss value for every 5 iterations


            # Gradient Descent
                

            # Compute gradient
            gradient = softmax_vector - np_one_hot_encode_target
            gradient /= np_data.shape[0] # Divide the gradient by the number of samples

            # print(f"The gradient is now {gradient}")

            # Compute gradients for weights and biases
            dW = np.dot(np_data.T, gradient)
            dB = np.sum(gradient, axis=0)
            # print(f"The value for dW is {dW} and the value for dB is {dB}")

            # Update weights and biases
            self.weights -= self.learning_rate * dW
            self.biases -= self.learning_rate * dB
            print(f"The new weights are {self.weights} and the new biases are {self.biases}")
            counter += 1
            print(f"The number of iterations completed is {counter}")

    def predict(self, data, legal=None):

        data = np.asarray(data) # Convert input array into numpy array
        print(f"Input data shape: {data.shape}")

        softmax_input = np.dot(data, self.weights) + self.biases
        print(f"Softmax input shape: {softmax_input.shape}")

        # Softmax function
        softmax_input = np.atleast_2d(softmax_input) # Ensures the logits are at least 2-dimensional
        exp_logits = np.exp(softmax_input - np.max(softmax_input, axis=1, keepdims=True))
        softmax_vector = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # Produces probabilities

        # softmax_probs = self.softmax(logits)
        print(f"Softmax probabilities shape: {softmax_vector.shape}")

        prediction = np.argmax(softmax_vector, axis=1)
        print(f"Prediction shape: {prediction.shape} and the prediction is {prediction}")

        return prediction

        

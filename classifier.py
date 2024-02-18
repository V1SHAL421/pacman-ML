# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np

class Classifier: # Use numpy and Multiclass Logistic Regression
    # Based off data, classifier should be able to choose which target (action) from 0-3
    def __init__(self):
        self.n_iterations = 100 # number of iterations
        self.learning_rate = 0.01
        self.weights = np.zeros((25, 4))
        self.biases = np.zeros(4)

    def softmax(self, logits):
        # Numerically stable softmax function
        logits = np.atleast_2d(logits)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        
        # Compute the cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def reset(self):
        self.weights = np.zeros_like(self.weights)
        self.biases = np.zeros_like(self.biases)
    
    def fit(self, data, target):
        # Convert input arrays into numpy arrays
        np_data = np.asarray(data)
        np_target = np.asarray(target)

        assert np_data.shape[1] == 25, f"Data shape mismatch. Expected 25 features, got {np_data.shape[1]}"
        
        np_one_hot_encode_target = np.eye(4)[np_target]
        # Ensure feature matrix and target vector are aligned
        print(f"Shape of the feature matrix is {np_data.shape} and the shape of the target vector is {np_target.shape}")

        counter = 0
        for iteration in range(self.n_iterations):

            softmax_input = np.dot(np_data, self.weights) + self.biases

            # Softmax function
            softmax_vector = self.softmax(softmax_input)

            # print(f"Softmax output vector is {softmax_vector}")

            # Cross Entropy Loss
            
            loss = self.cross_entropy_loss(np_one_hot_encode_target, softmax_vector)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {loss}")

            # Compute gradient
            gradient = softmax_vector - np_one_hot_encode_target
            gradient /= np_data.shape[0] # Divide the gradient by the number of samples

            # print(f"The new gradient is {gradient}")

            # Copmute gradients for weights and biases
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
        data = np.asarray(data)
        print(f"Input data shape: {data.shape}")
        logits = np.dot(data, self.weights) + self.biases
        print(f"Logits shape: {logits.shape}")
        softmax_probs = self.softmax(logits)
        print(f"Softmax probabilities shape: {softmax_probs.shape}")
        prediction = np.argmax(softmax_probs, axis=1)
        print(f"Prediction shape: {prediction.shape}")
        return prediction

        

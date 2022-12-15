# Importing libraries for the motherfucker
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import optimize
import math
import utils
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat


#Load the motherfucker
filename = 'data/encodedShroomsV2.csv'
df = pd.read_csv(filename)
#print(df)

# initializing datasets, filling missing values with zeroes
train_size = 2 #5700
X = np.empty((train_size, 109), dtype=int)
y = []
X_test = np.empty((8124-train_size, 109), dtype=int)
y_test = []
j = 0
for column in df:
    if j < train_size:
        if len(df[column].name) == 110:
            X_row = []
            if df[column].name[0] == '1':
                y.append(1)
            else:
                y.append(0)
        
            for i in range(110):
                if i > 0:
                    if df[column].name[i] == '1':
                        X_row.append(1)
                    else:
                        X_row.append(0)
            X[j] = X_row
        if len(df[column].name) == 112:
            X_row = []
            if df[column].name[0] == '1':
                y.append(1)
            else:
                y.append(0)
        
            for i in range(110):
                if i > 0:
                    if df[column].name[i] == '1':
                        X_row.append(1)
                    else:
                        X_row.append(0)
            X[j] = X_row
    else:
        if len(df[column].name) == 110:
            X_row = []
            if df[column].name[0] == '1':
                y_test.append(1)
            else:
                y_test.append(0)
        
            for i in range(110):
                if i > 0:
                    if df[column].name[i] == '1':
                        X_row.append(1)
                    else:
                        X_row.append(0)
            X_test[j-train_size] = X_row
        if len(df[column].name) == 112:
            X_row = []
            if df[column].name[0] == '1':
                y_test.append(1)
            else:
                y_test.append(0)
        
            for i in range(110):
                if i > 0:
                    if df[column].name[i] == '1':
                        X_row.append(1)
                    else:
                        X_row.append(0)
            X_test[j-train_size] = X_row
    j = j+1
        
y = np.array(y)
y_test = np.array(y_test)


#22 input layers, 2 output layers
# Setup the parameters you will use for this exercise
input_layer_size  = 110  # 22 features
hidden_layer_size = 5   # 5 hidden units
num_labels = 1          # 1 label, poisonous/edible

# Load the weights into variables Theta1 and Theta2

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = np.ones((5,110)),np.ones((1,6))

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing


# Unroll parameters 
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network.
    
    Parameters
    ----------
    Theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)
    
    Theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)
    
    X : array_like
        The image inputs having shape (number of examples x image dimensions).
    
    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned neural
    network. You should set p to a vector containing labels 
    between 0 to (num_labels-1).
     
    Hint
    ----
    This code can be done all vectorized using the numpy argmax function.
    In particular, the argmax function returns the index of the  max element,
    for more information see '?np.argmax' or search online. If your examples
    are in rows, then, you can use np.argmax(A, axis=1) to obtain the index
    of the max for each row.
    
    Note
    ----
    Remember, we have supplied the `sigmoid` function in the `utils.py` file. 
    You can use this function by calling `utils.sigmoid(z)`, where you can 
    replace `z` by the required input variable to sigmoid.
    """
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    a2 = utils.sigmoid(X.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    p = np.argmax(utils.sigmoid(a2.dot(Theta2.T)), axis = 1)


    # =============================================================
    return p

def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .
    
    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.
    
    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 
    
    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned logistic
    regression parameters (one-vs-all). You should set p to a vector of predictions
    (from 0 to num_labels-1).
    
    Hint
    ----
    This code can be done all vectorized using the numpy argmax function.
    In particular, the argmax function returns the index of the max element,
    for more information see '?np.argmax' or search online. If your examples
    are in rows, then, you can use np.argmax(A, axis=1) to obtain the index 
    of the max for each row.
    """
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    p = np.argmax(utils.sigmoid(X.dot(all_theta.T)), axis = 1)


    
    # ============================================================
    return p


def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network.
    
    Parameters
    ----------
    Theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)
    
    Theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)
    
    X : array_like
        The image inputs having shape (number of examples x image dimensions).
    
    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned neural
    network. You should set p to a vector containing labels 
    between 0 to (num_labels-1).
     
    Hint
    ----
    This code can be done all vectorized using the numpy argmax function.
    In particular, the argmax function returns the index of the  max element,
    for more information see '?np.argmax' or search online. If your examples
    are in rows, then, you can use np.argmax(A, axis=1) to obtain the index
    of the max for each row.
    
    Note
    ----
    Remember, we have supplied the `sigmoid` function in the `utils.py` file. 
    You can use this function by calling `utils.sigmoid(z)`, where you can 
    replace `z` by the required input variable to sigmoid.
    """
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    a2 = utils.sigmoid(X.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    p = np.argmax(utils.sigmoid(a2.dot(Theta2.T)), axis = 1)


    # =============================================================
    return p

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

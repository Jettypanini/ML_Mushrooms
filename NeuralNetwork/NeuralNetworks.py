# Importing libraries for the motherfucker
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import optimize
import math
import utils
import matplotlib.pyplot as plt
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat


#Load the motherfucker
filename = 'data/encodedShroomsV2NoDupes.csv'
df = pd.read_csv(filename)
#print(df)

# initializing datasets, filling missing values with zeroes
train_size = 3000 #300 = 88.7%, 400 = 89.5%, 500 = 90.4%, 550 = 90.4% 600 = 89.7%, 1000 = 89.8, 2000 = 88.1,  3000 = 90.3%
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
            print("112")
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
            print("112")
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
input_layer_size  = 109  # 22 features
hidden_layer_size = 25  # 5 hidden units
num_labels = 2          # 2 labels, poisonous/edible

# Load the weights into variables Theta1 and Theta2
Theta1, Theta2 = np.ones((25,110)), np.ones((2,26))
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

#print('Initializing Neural Network Parameters ...')
# Theta1 has size 25 x 110
# Theta2 has size 2 x 26
initial_Theta1 = utils.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = utils.randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

utils.checkNNGradients(utils.nnCostFunction)

#  Check gradients by running checkNNGradients
lambda_ = 3
utils.checkNNGradients(utils.nnCostFunction, lambda_)

# Also output the costFunction debugging values
debug_J, _  = utils.nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X, y, lambda_)

#print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
#print('(for lambda = 3, this value should be about 0.576051)')

#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options= {'maxiter': 100}

#  You should also try different values of lambda
lambda_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: utils.nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# get the solution of the optimization
nn_params = res.x
        
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

pred = utils.predict(Theta1, Theta2, X)
f, axarr = plt.subplots(2,1)
axarr[0].imshow(Theta1, cmap='binary', interpolation='none')
axarr[1].imshow(Theta2, cmap='binary', interpolation='none')
plt.show()
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))








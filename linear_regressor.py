'''

PYTHON MACHINE LEARNING COOKBOOK -- PRATEEK JOSHI
CHAPTER 1 - THE REALM OF SUPERVISED LEARNING

'''

import sys
import numpy as np

filename = sys.argv[1]
x = []  # data
y = []  # data

with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

num_training = int(0.8 * len(x))
num_test = len(x) - num_training

x_train = np.array(x[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])

x_test = np.array(x[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])

from sklearn import linear_model

# create linear regression object
regressor = linear_model.LinearRegression()
# train the model using the training data
regressor.fit(x_train, y_train)

import matplotlib.pyplot as plt

y_train_pred = regressor.predict(x_train)
plt.figure()
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

y_test_pred = regressor.predict(x_test)
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, y_train_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

'''
Computing error.
Error is defined as the difference between the actual value and the value predicted by the regressor

1. Mean absolute error(MAE): average of all absolute error of the datapoints of a given dataset
2. Mean squared error(MSE): average of the squares of the error of all the datapoints in the dataset
3. Median absolute error: median of all errors in the dataset. It is robust to outliers which means
                        a single bad point in the dataset wouldn't skew the entire error metrics
4. Explained variance score: measures how well our model can account for the variation in our dataset.
                            A score of 1.0 indicates that our model is perfect
5. R2 score: coefficient of determination. It tells us how well unknown samples will be predicted by our model

A good practice is to make sure the mean squared error is low and explained variance score is high
'''

import sklearn.metrics as sm
print("Mean Absolute Error = ", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean Squared Error = ", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median Absolute Error = ", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score = ", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred), 2))


import pickle as pickle

output_model_file = 'saved_model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(x_test)
print("\nNew mean absolute error = ", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))


'''
Ridge regressor uses regularization where a penalty is imposed on the size of the coefficients

https://www.mygreatlearning.com/blog/what-is-ridge-regression/
It is a model tuning method that is used to analyse any data that suffers from multicollinearity
It performs L2 regularization. 
'''












'''
Code to run script
python linear_regressor.py data_singlevar.txt

'''
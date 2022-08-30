'''

PYTHON MACHINE LEARNING COOKBOOK -- PRATEEK JOSHI
CHAPTER 1 - THE REALM OF SUPERVISED LEARNING

'''

import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])
print(data)

# data preprocessing
'''
Mean Removal : It is beneficial to remove the mean from each feature so that
it is centered on zero. Helps in removing bias from the features.

Questions:
1. What does 'centered on zero mean?
 
https://www.scribbr.com/statistics/mean/
https://www.scribbr.com/statistics/standard-deviation/
https://www.quora.com/What-is-zero-centering-data-preprocessing-technique
It’s a linear transformation of the data that shifts the data so it’s centered at the origin. 
Usually this is done by subtracting the mean vector from every data point. It’s done to find 
the “natural coordinate system” for the data. An extension of this is scaling so the data is 
not just centered at the origin, but the standard deviation is normalized to one by scaling. 
Even beyond this, principal component analysis which aims to transform the coordinate system 
not just so that it is origin centered, but so that the primary components of independent 
variation lie on the different axis. The reason to do this is that it makes the data much easier to work with.

2. What is bias?

https://www.bmc.com/blogs/bias-variance-machine-learning/
Bias is a phenomenon that skews the result of an algorithm in favor or against an idea.
Bias is considered a systematic error that occurs in the machine learning model itself 
due to incorrect assumptions in the ML process.Technically, we can define bias as the 
error between average model prediction and the ground truth.
'''

data_standardized = preprocessing.scale(data)
print("\nMean = ", data_standardized.mean(axis=0))
print("STD deviation = ", data_standardized.std(axis=0))

'''
Scaling allow values of each feature in a datapoint to be in the same playing field
'''
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data = ", data_scaled)

'''
Data Normalization is used to adjust the values in the feature vector so they can 
be measured on a common scale 
'''

data_normalized = preprocessing.normalize(data, norm='l1')
print("\nL1 normalized data = ", data_normalized)

'''
Binarization is used to convert numerical feature vector into boolean vector
'''

data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\nBinarized data = ", data_binarized)

'''
One Hot Encoding comes to play when we deal with numerical values that are
sparse and scattered all over the place. We can look at One-Hot encoding as
a tool to tighten the feature vector. It looks at each feature and identifies 
the total number of distinct values.
'''

encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\nEncoded vector = ", encoded_vector)

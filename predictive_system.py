# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/tonya/ML_Deploy/Prediction_Model/models/diabetes_trained_model.sav', 'rb'))

input_data = (3,126,88,41,235,39.3,0.704,27)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

scaler = StandardScaler()
# standardize the input data
std_data = scaler.fit_transform(input_data_reshaped)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
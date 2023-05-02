# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:38:15 2023

@author: tonya
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

Diabetes_trained_model = pickle.load(open('C:/Users/tonya/ML_Deploy/Prediction_Model/models/diabetes_trained_model.sav', 'rb'))
heart_trained_model = pickle.load(open('C:/Users/tonya/ML_Deploy/Prediction_Model/models/heart_trained_model.sav', 'rb'))
parkinson_trained_model = pickle.load(open('C:/Users/tonya/ML_Deploy/Prediction_Model/models/parkinson_trained_model.sav', 'rb'))


# Sidebar for Navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
# Creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    scaler = StandardScaler()
    # standardize the input data
    std_data = scaler.fit_transform(input_data_reshaped)

    prediction = Diabetes_trained_model.predict(std_data)
    # print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    # Giving Title
    st.title('Diabetes Predction Web App')
    
    # Getting input data from the user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Level')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of the Person')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
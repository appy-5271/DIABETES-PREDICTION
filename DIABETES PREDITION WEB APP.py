# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:01:42 2024

@author: Lenovo
"""

import numpy as np 
import pickle 
import streamlit as st 

#Loading the saved model 
loaded_model = pickle.load(open('C:/Users/Lenovo/Downloads/ML DEPLOY/trained_model.sav', 'rb'))

#Creating a function for prediction 

def diabetes_prediction(input_data):
    
    #changing the input data to numpy array
    input_data_to_numpyarray = np.asarray(input_data)

    #RESAHPE THE ARRAY AS WE ARE PREDICTING FOR ONLY ONE INSTANCE
    input_data_reshaped = input_data_to_numpyarray.reshape(1,-1)


    #NOW WE NEED TO STANDARDINZE THIS DATA AS THE MODEL WORKS ON STANDARD DATA
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0) :
      return"THE PERSON IS NOT DIABETIC"
    else:
      return "THE PERSON IS DIABETIC"
    
    
def main():
        
        
        #Giving a titel for the the web page 
        st.title('DIABETES PREDICTION WEB APP')
        
        #GETTING the input data from the user 
        #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
        
        Pregnancies = st.text_input('Number of pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure  = st.text_input('Blood Pressure Value')
        SkinThickness= st.text_input('Skin Thickness')
        Insulin = st.text_input('Enter the insulin value')
        BMI = st.text_input('ENter BMI value')
        DiabetesPedigreeFunction = st.text_input('Enter function value')
        Age = st.text_input('Age')
        
        
        #CODe for prediction 
        diagnosis = ''
        
        #creating a button for prediction 
        
        if st.button('Diabetes test result'):
            diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin, BMI,DiabetesPedigreeFunction,Age])
       
        st.success(diagnosis)
        
        
        
if __name__ == '__main__':
     main()
        
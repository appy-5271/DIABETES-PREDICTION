import numpy as np
import pickle 

#Loading the saved model 

loaded_model = pickle.load(open('C:/Users/Lenovo/Downloads/ML DEPLOY/trained_model.sav', 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)


#changing the input data to numpy array
input_data_to_numpyarray = np.asarray(input_data)

#RESAHPE THE ARRAY AS WE ARE PREDICTING FOR ONLY ONE INSTANCE
input_data_reshaped = input_data_to_numpyarray.reshape(1,-1)


#NOW WE NEED TO STANDARDINZE THIS DATA AS THE MODEL WORKS ON STANDARD DATA
#std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0]==0 :
  print("THE PERSON IS NOT DIABETIC")
else:
  print("THE PERSON IS DIABETIC")
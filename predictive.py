import numpy as np
import pickle

#loading the saved model

loaded_model=pickle.load(open("C:/Users/kakul/OneDrive/Desktop/streamlit/Admission task 1/Admission_model.sav",'rb'))

input_data=(337,118,4,4.5,4.5,9.65,1)
input_data_as_numpy_array=np.array(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print("The person has higher chance of getting graduation admission")
else:
     print("The person has little or no chance of getting graduation admission")
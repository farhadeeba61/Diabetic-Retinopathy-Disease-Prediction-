#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd


# In[2]:


from pickle import load


# In[3]:


model=load(open('log_reg.pkl','rb'))
scale=load(open('scaler.pkl','rb'))


# ### App title

# In[4]:


st.title("PREDICTION OF DIABETIC RETINOPATHY")
st.write("Enter Patient Data below:")


# In[7]:


age =st.number_input('Age',min_value =0,max_value= 120, value=30)
systolic_bp  = st.number_input('systolic Blood Pressure',min_value =50,max_value= 250, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure',min_value =30,max_value= 150, value=80)
cholesterol = st.number_input('Cholesterol',min_value =100 ,max_value= 400, value=200)


# ### Predict button

# In[8]:


if st.button('Predict'):
    #### giving input data
    input_data = pd.DataFrame(
        [[ age, systolic_bp , diastolic_bp, cholesterol]],
        columns=[' age', ' systolic_bp ', ' diastolic_bp', ' cholesterol']
    )
    input_scaled =scale.transform(input_data)
    prediction = model.predict(input_scaled) [0]
    probability= model.predict_proba(input_scaled)[0][1]
    st.write(prediction)
    ##### Displaying results
    st.subheader("Prediction Results")
    st.write(f" Predicted_Class: {prediction} (0=No Retinopathy,1= Retinopathy)")
    st.write(f" Probability of Retinopathy: {probability: .2f}")
    ####Displaying the input data for reference
    st.write(" Entered Patient Data")
    st.write(input_data)


# In[ ]:





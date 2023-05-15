import streamlit as st 
import numpy as np 
import pandas as pd  
import joblib 
from feature_engineering_app import feature_engineering_app


st.title("BIKE SHARING APP ") 
st.write('**this app for predicting number of :blue[casual] and :blue[registered] users per _Hour_  :sunglasses:**')

st.write("\n")
st.write("\n")


col1, col2 = st.columns(2)

with col1 :
    temp=st.number_input('Temperature :thermometer:',min_value=0,max_value=100)
    humidity=st.number_input('Humidity :sweat_drops:',min_value=0,max_value=100)
    weather=st.selectbox('Weather :partly_sunny:',options=['Clear','Cloudy','Snow']) 
    
    

with col2 : 
    windspeed=st.number_input('Windspeed :dash:',min_value=0,max_value=100)
    season=st.selectbox('Season :leaves:',options=['summer','winter','fall','spring'])
    day=st.selectbox('Day :smile:',options=['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']) 


month=st.selectbox('Month :date:',options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']) 
st.write('\n') 
is_holiday=st.checkbox('Is holiday ')
if is_holiday : 
     is_holiday=1 
else : 
    is_holiday=0


is_workingday=st.checkbox('Is workingday')
if is_workingday : 
   is_workingday=1 
else : 
    is_workingday=0
st.write('\n')


hour=st.slider('Hour :clock9: ',0,23)

st.write('\n')
submit=st.button('**Predict**') 

choices=[temp,humidity,windspeed,season,is_holiday,is_workingday,weather,hour,month,day]


feature_engineering=feature_engineering_app()
preprocessing=joblib.load('preprocessor.pickle') 
modeling=joblib.load('model.pickle')

if submit :  
    fe_data=feature_engineering.transform([choices])
    prc_data=preprocessing.transform(fe_data)
    mdl_data=modeling.predict(prc_data) 
    st.write('you will have ',round(mdl_data[0][0]),'casual users')
    st.write('you will have ',round(mdl_data[0][1]),'registered users')

import streamlit as st
import numpy as np
import pandas as pd
from pickle import load

st.title("Predict the diamond price")

# here we define some of the front end elements of the web page like
# the font and background color, the padding and the text to be displayed
html_temp = """
    <div style ="background-color:black;padding:13px">
    <h1 style ="color:white;text-align:center;"> Machine learning Application for  Predicting the Diamond Price  </h1>
    </div>
    <h2> Enter the details about your diamond to know its price </h2>

    """

# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html=True)

scaler = load(open('E__diamond price app_standard_scaler.pkl', 'rb'))


rf_model = load(open('E__diamond price app_randomforestregression.pkl', 'rb'))

label_cut = {'Ideal':2, 'Premium':3, 'Very Good':4, 'Good':1, 'Fair':0}
label_color = {'G':3, 'E':1, 'F':2, 'H':4, 'D':0, 'I':5, 'J':6}
label_clarity = {'SI1':2, 'VS2':5, 'SI2':3, 'VS1': 4, 'VVS2':7, 'VVS1':6, 'IF':1, 'I1':0}
carat = st.slider("Carat", 0.2, 5.01)
table = st.slider("Width", 43.00, 95.00)
depth = st.slider("Depth", 43.00, 79.00)
x = st.slider('X',0.00,10.70)
y = st.slider('Y',0.00,58.90 )
z = st.slider('Z',0.00,31.80 )

cut = st.selectbox(
    'How would be the cut of Diamond?',
    ('Fair', 'Good', 'Very Good', 'Ideal', 'Premium'))

color = st.selectbox(
    'What should be the color of Diamond?',
    ('J', 'I', 'H', 'G', 'F', 'E', 'D'))

clarity = st.selectbox(
    'How would you like to be contacted?',
    ('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))

btn_click = st.button("Predict")

if btn_click == True:
    if carat and depth and table and x and y and z:

        query_point_num_transformed = scaler.transform(
            [[float(carat), float(depth), float(table), float(x), float(y), float(z)]])
        query_point_cat = np.array([label_clarity[clarity], label_color[color], label_cut[cut]])  # .reshape(1,-1)

        df = np.concatenate((query_point_cat, query_point_num_transformed.flatten()), axis=None)
        pred = rf_model.predict(df.reshape(1, -1)).item()
        st.success(pred)
        st.balloons()
    else:
        st.error('Enter the values properly')
        st.snow()
import streamlit as st
import pandas as pd
import numpy as np


df1 = pd.read_csv("open_pubs.csv")
df1.columns=['fsa_id','name','address','postcode','easting','northing','lat','lon','local_authority']
df1=df1.replace('\\N',np.NaN)
df1=df1.dropna()
df1.head(100)
df1[['lat', 'lon']] = df1[['lat', 'lon']].apply(pd.to_numeric)
df1.info()


st.markdown('<h1 style="color:green">UK MAP PUB Locations</h1>', unsafe_allow_html=True)


local = df1["local_authority"].unique()
st.sidebar.markdown("Chosing the Area")
option = st.sidebar.selectbox('',local)


'You Selected : ' ,option



btn_clk = st.button('Search')
if btn_clk==True:
    res = df1[df1['local_authority']==option]
    res=res[['lat','lon']]
    st.success('It display\'s all the pubs in the area that are selected')
    st.map(res)
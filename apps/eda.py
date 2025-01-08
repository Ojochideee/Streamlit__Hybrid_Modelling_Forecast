import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'apps'))

from data import load_ireland_data, load_uk_data  # Import functions from data.py

def app():
    # Web App Title
    st.markdown('''
    # **Exploratory Data Analysis (EDA)**

    Welcome to the **EDA Section** of our Inflation Prediction App! Here, 
                you'll dive deep into the macroeconomic datasets of **Ireland** and the **UK**. 
                Using advanced exploratory tools, including the powerful **pandas-profiling** library.

    Explore key features such as:
    - Inflation rates
    - Exchange rates
    - Housing market dynamics
    - Immigration trends
    - GDP 
    - Unemployment rates
    - Interest rates
    - And Close price of the stock market

    Gain a comprehensive understanding of the data that drives our hybrid prediction model and discover the patterns that influence economic indicators.

    ---
    ''')


    # Sidebar for dataset selection
    st.sidebar.header('Dataset Selection')
    dataset_choice = st.sidebar.selectbox('Choose a Country', ['Ireland', 'United Kingdom'])

    if dataset_choice == 'Ireland':
        df = load_ireland_data()
        st.header('**Ireland Dataset**')
        st.write(df)
        pr = ProfileReport(df, explorative=True)
        st.header('**Pandas Profiling Report for Ireland**')
        st_profile_report(pr)
    else:
        df = load_uk_data()
        st.header('**United Kingdom Dataset**')
        st.write(df)
        pr = ProfileReport(df, explorative=True)
        st.header('**Pandas Profiling Report for United Kingdom**')
        st_profile_report(pr)
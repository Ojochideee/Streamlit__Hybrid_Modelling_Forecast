import streamlit as st
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'apps'))
from data import load_ireland_data, load_uk_data

def app():
    # Header with icon
    col1, col2 = st.columns([1, 3])
    icon_path = "/Users/blueaivee/Desktop/Streamlit_hybrid_devop/Datasets/graph_icon.png"

    with col1:
        if os.path.exists(icon_path):
            st.image(icon_path, width=80)
        else:
            st.write("Icon not found")

    with col2:
        st.title('Inflation Prediction App')

    # Interactive Welcome Message
    st.markdown("""
        ## Welcome!
        Explore macroeconomic data and predict inflation with our advanced **Hybrid Model**.
        """)

    # User Input Section
    st.write("### Customize Your Analysis")
    country = st.selectbox("Select Country", ["Ireland", "UK"])

    # Load dataset based on selected country
    if country == "Ireland":
        data = load_ireland_data()
    else:
        data = load_uk_data()

    # Quick Statistics
    st.write("### Key Insights")
    current_inflation_rate = data['Inflation rate'].iloc[-1]
     # Assuming the last row is the most recent prediction
    st.metric("Current Inflation Rate", f"{current_inflation_rate:.1f}%")

    # Feature Selection for Plotting
    st.write("### Recent Trends")
    feature = st.selectbox("Select Feature to Plot", data.columns.drop('Date'))
    st.line_chart(data.set_index('Date')[feature])

    # Navigation Guide
    st.write("""
        Use the sidebar to navigate through different sections:
        - **EDA**: Explore the dataset
        - **Hybrid Model**: View the prediction models
        """)

    # Footer
    st.write("---")
    st.write("Developed by Tonia Ameh | Contact: anthoniaameh92@gmail.com")

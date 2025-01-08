import streamlit as st
from multiapp import MultiApp
from apps import home, eda, hybrid_model

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Inflation Rate Prediction', page_icon=':bar_chart:',
    layout='wide')

#---------------------------------#

app = MultiApp()

# Add all your application pages here
app.add_app("Home", home.app)
app.add_app("EDA", eda.app)
app.add_app("Modelling", hybrid_model.app)

# The main app
app.run()


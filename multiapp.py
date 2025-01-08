# Description: This file contains the MultiApp class which is used to combine multiple streamlit applications into a single app.
import streamlit as st

class MultiApp:
    #Framework for combining multiple streamlit applications.
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.selectbox(
            'Explore the App',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()
import streamlit as st
import pandas as pd

import numpy as np
import joblib
import re

def show_common_page():
    # st.write(""" Hello""")

    with open('./files/style.css') as f:
        css = f.read()

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
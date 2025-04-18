import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import streamlit.components.v1 as components

from common_page import show_common_page

show_common_page()

# Load the pre-trained model pipeline and data
model = joblib.load('overcrowding_model_pipeline.pkl')

data = pd.read_csv('tourist_attraction.csv')

# Replace "Not available" or similar placeholders with NaN
data.replace(["Not available", "Closed"], np.nan, inplace=True)
# Convert visitor count columns to numeric (removing commas)
visitor_cols = ['Visitors_2018', 'Visitors_2019', 'Visitors_2020', 'Visitors_2021', 'Visitors_2022', 'Visitors_2023']
for col in visitor_cols:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')

def categorize_overcrowding(visitors_2023, bracket):
    # Ensure visitors_2023 is numeric and not NaN
    if pd.notna(visitors_2023):
        if visitors_2023 > 100000:
            return 'High'
        elif visitors_2023 > 20000:
            return 'Medium'
        else:
            return 'Low'
    else:
        if bracket in ["Over 200,000", "100,001-200,000"]:
            return 'High'
        elif bracket in ["50,001-100,000", "20,001-50,000"]:
            return 'Medium'
        elif bracket in ["10,001 to 20,000", "10,000 or less"]:
            return 'Low'
        else:
            return None

data['Overcrowding_Level'] = data.apply(
    lambda row: categorize_overcrowding(row['Visitors_2023'], row['Visitor admission brackets']), axis=1
)

data.dropna(subset=['Overcrowding_Level'], inplace=True)


# from IPython.display import IFrame

# Display the HTML map in the notebook
# IFrame(src='uk_overcrowding_map.html', width='100%', height=600)

st.header("Overall Overcrowding Levels Map")

HtmlFile = open("uk_overcrowding_map.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code, height = 600)

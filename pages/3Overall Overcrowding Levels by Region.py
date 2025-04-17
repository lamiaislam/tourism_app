import streamlit as st
import pandas as pd

import numpy as np
import joblib
import re
import matplotlib.pyplot as plt


from common_page import show_common_page

show_common_page()


# Load the pre-trained model pipeline and data
model = joblib.load('overcrowding_model_pipeline.pkl')
# data = pd.read_csv('tourist_attraction.csv').replace("Not available", pd.NA)

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




# Main panel: Data Visualizations
st.subheader("Overall Overcrowding Levels by Region")
# Calculate percentage of attractions in each overcrowding category per region
region_pivot = pd.crosstab(data['Region'], data['Overcrowding_Level'], normalize='index') * 100
st.dataframe(region_pivot.style.format("{:.1f}%"))  # show as percentage table

# Bar chart: Total visitors by region (as another indicator of crowding)
region_visitors = data.groupby('Region')['Visitors_2023'].sum().reset_index()
region_visitors = region_visitors.dropna()  # drop regions with no data
st.bar_chart(region_visitors.set_index('Region')['Visitors_2023'])







st.subheader("📊 Histogram: Number of Attractions by Category")

# Count number of attractions by Category
category_counts = data['Category'].value_counts()

# Matplotlib plot in Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
category_counts.plot(kind='bar', color='lightgreen', ax=ax)
ax.set_title("Number of Attractions by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Number of Attractions")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

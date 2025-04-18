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

# Clean/replace if necessary
data.replace("Not available", pd.NA, inplace=True)

# Prepare region-level count
region_counts = data['Region'].value_counts()

# Streamlit section
st.subheader("📊 Region-wise crowded percentage")

# Plot with matplotlib
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
ax.set_title("Attraction Distribution by Region")
ax.axis("equal")

# Display in Streamlit
st.pyplot(fig)









# Streamlit App Title
st.subheader("🌍 Area Type Distribution (CRU)")

# Count occurrences of CRU types (Coastal, Rural, Urban)
cru_counts = data['CRU'].value_counts()

# Plot pie chart
fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(cru_counts, labels=cru_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
ax.set_title("Distribution of Area Types (CRU: Coastal, Rural, Urban)")
ax.axis('equal')  # Equal aspect ratio makes pie a circle

# Display the pie chart in Streamlit
st.pyplot(fig)

# Optionally display the counts as a table for reference
st.write("CRU Distribution Table:")
st.dataframe(cru_counts)



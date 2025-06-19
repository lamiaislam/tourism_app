import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime

st.set_page_config(
        page_title="Tourist Tracker",
)

from common_page import show_common_page

show_common_page()

# Load the trained model pipeline
model = joblib.load("overcrowding_model_pipeline.pkl")

# Rebuild the pipeline in code instead of loading
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define numeric processing
numeric_features = ['Visitors_2019', 'Visitors_2021', 'Visitors_2022']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, eval_metric='mlogloss', use_label_encoder=False))
])

# Now fit or use your pipeline here
# pipeline.fit(X_train, y_train)







# Load dataset
data = pd.read_csv("tourist_attraction.csv").replace("Not available", np.nan)

# Clean numeric columns
visitor_cols = ['Visitors_2019', 'Visitors_2021', 'Visitors_2022']
for col in visitor_cols:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')

# Streamlit UI
st.title("🎯 Tourist Overcrowding Prediction Display")
st.markdown("Choose destination details and check predicted **overcrowding level**!")


# Add options for Region, District, Category, and Attraction
region = st.selectbox("📍 Select Region", sorted(data['Region'].dropna().unique()))
district = st.selectbox("🏙 Select District", sorted(data[data['Region'] == region]['District'].dropna().unique()))
category = st.selectbox("🏛️ Select Category", sorted(data[(data['Region'] == region) & (data['District'] == district)]['Category'].dropna().unique()))
attraction = st.selectbox("🎡 Select Attraction", sorted(data[(data['Region'] == region) & (data['District'] == district) & (data['Category'] == category)]['Attraction'].dropna().unique()))

# User selects the future date for prediction
future_date = st.date_input("📅  Select Date", min_value=datetime.today())

# Display selected date
st.write(f"Selected Date: {future_date}")

# Get attraction data
attraction_data = data[data['Attraction'] == attraction].iloc[0]

# Get the previous visitor data for that attraction (2019, 2021, 2022)
visitors_2019 = attraction_data['Visitors_2019'] if pd.notna(attraction_data['Visitors_2019']) else data['Visitors_2019'].median()
visitors_2021 = attraction_data['Visitors_2021'] if pd.notna(attraction_data['Visitors_2021']) else data['Visitors_2021'].median()
visitors_2022 = attraction_data['Visitors_2022'] if pd.notna(attraction_data['Visitors_2022']) else data['Visitors_2022'].median()

# Area Type (CRU) and Charging Fee
area_type = attraction_data['CRU']
charging = attraction_data['Charging']

# Prepare the data for the model prediction
input_df = pd.DataFrame({
    "Region": [region],
    "Category": [category],
    "CRU": [area_type],
    "Charging": [charging],
    "Visitors_2019": [visitors_2019],
    "Visitors_2021": [visitors_2021],
    "Visitors_2022": [visitors_2022]
})

# Make prediction when user clicks the button
if st.button("🔮 Predict Overcrowding"):
    prediction = model.predict(input_df)[0]
    
    st.subheader(f"Prediction Result for {future_date}: **{prediction}**")
    if prediction == "High":
        st.error("⚠️ Expect very heavy crowding.")
        # (Alternative suggestions logic here)
    elif prediction == "Medium":
        st.warning("⚠️ Explore other times.")
    else:
        st.success("✅ Safe to be visited.")

if 'prediction' in locals():
    # Suggest alternatives if overcrowding is high
    if prediction == "High":
        
        st.html(
            "<a href='#suggested-alt' style='color:lawngreen;'><h4>🧭 Suggested Alternatives<h4></a>"
        )

# Suggest alternatives if overcrowding is high

if 'prediction' in locals():
    if prediction == "High":
        
        with st.sidebar:
            st.html(
                "<a href='#suggested-alt' style='color:lawngreen;'><h4>🧭 Suggested Alternatives<h4></a>"
            )
        st.markdown("---")
        
        st.html(
            "<h3 id='suggested-alt'>🧭 Suggested Alternatives with Lower Crowding<span data-testid='stHeaderActionElements' class='st-emotion-cache-gi0tri e121c1cl3'><a href='#77d7bf98' class=st-emotion-cache-ubko3j e121c1cl1'><svg xmlns='http://www.w3.org/2000/svg' width='1rem' height='1rem' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M15 7h3a5 5 0 0 1 5 5 5 5 0 0 1-5 5h-3m-6 0H6a5 5 0 0 1-5-5 5 5 0 0 1 5-5h3'></path><line x1='8' y1='12' x2='16' y2='12'></line></svg></a></span></h3>"
        )                                                                   
        # Filter for same category, different attraction, low crowding, similar region
        alternatives = data[
            (data['Category'] == category) &
            (data['Attraction'] != attraction) &
            (data['Region'] == region) &
            (data['Visitors_2023'].notna())  # ensure we have 2023 data to compare
        ].copy()
        
        # Convert visitor column properly
        alternatives['Visitors_2023'] = pd.to_numeric(alternatives['Visitors_2023'], errors='coerce')
        
        # Filter for low crowded alternatives based on 2023 visitor threshold
        low_crowd_alts = alternatives[alternatives['Visitors_2023'] <= 20000]
        
        # Show top 3 based on lowest visitors
        if not low_crowd_alts.empty:
            suggested = low_crowd_alts.sort_values(by='Visitors_2023').head(3)

            for idx, row in suggested.iterrows():
                st.markdown(f"### 🎡 {row['Attraction']}")
                st.write(f"📍 **Region**: {row['Region']}")
                st.write(f"🏙 **District**: {row['District']}")
                st.write(f"🏛 **Category**: {row['Category']}")
                st.write(f"🌍 **Area Type (CRU)**: {row['CRU']}")
                st.write(f"💷 **Charges Entry Fee?**: {row['Charging']}")
                st.write(f"📊 **Visitors in 2023**: {int(row['Visitors_2023'])}")
                st.markdown("---")
        else:
            st.info("No low-crowded alternatives found in the same category and region.")






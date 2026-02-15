import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "shivkumark0911gl/tourism-wellness-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="best_rf_model.pkl",
    repo_type="model"
)
model = joblib.load(model_path)

st.set_page_config(page_title="Wellness Tourism Package Prediction", layout="centered")
st.title("Wellness Tourism Package Purchase Prediction")

age = st.number_input("Age", 18, 80, 30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
num_followups = st.number_input("Number of Followups", 0, 10, 2)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", 0, 50, 2)
passport = st.selectbox("Passport (0 = No, 1 = Yes)", [0, 1])
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Own Car (0 = No, 1 = Yes)", [0, 1])
num_children = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.text_input("Designation", "Executive")
monthly_income = st.number_input("Monthly Income", 1000, 200000, 50000)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

for col in input_df.select_dtypes(include="object").columns:
    input_df[col] = input_df[col].astype("category").cat.codes

if st.button("Predict Purchase Probability"):
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted Probability of Purchase: {probability:.2f}")
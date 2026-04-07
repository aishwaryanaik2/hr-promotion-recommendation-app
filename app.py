import streamlit as st
import joblib
import pandas as pd

# Load top 3 models and feature columns
models = joblib.load("top3_hr_models.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("HR Promotion Recommendation App")

# Top 3 model dropdown
selected_model_name = st.selectbox(
    "Select Best Model",
    list(models.keys())
)

model = models[selected_model_name]

st.subheader("Enter Employee Details")

# Clean original user-friendly inputs
department = st.selectbox(
    "Department",
    ["Sales & Marketing", "Operations", "Technology", "Analytics", "Finance", "HR"]
)

region = st.selectbox(
    "Region",
    [f"region_{i}" for i in range(1, 35)]
)

education = st.selectbox(
    "Education",
    ["Below Secondary", "Bachelor's", "Master's & above"]
)

gender = st.selectbox("Gender", ["m", "f"])

recruitment_channel = st.selectbox(
    "Recruitment Channel",
    ["sourcing", "referred", "other"]
)

no_of_trainings = st.number_input("No of Trainings", min_value=0)
age = st.number_input("Age", min_value=18)
previous_year_rating = st.number_input(
    "Previous Year Rating",
    min_value=0,
    max_value=5
)
length_of_service = st.number_input("Length of Service", min_value=0)
KPIs_met = st.selectbox("KPIs > 80%", [0, 1])
awards_won = st.selectbox("Awards Won", [0, 1])
avg_training_score = st.number_input("Average Training Score", min_value=0)

# Build original dataframe
input_df = pd.DataFrame([{
    "department": department,
    "region": region,
    "education": education,
    "gender": gender,
    "recruitment_channel": recruitment_channel,
    "no_of_trainings": no_of_trainings,
    "age": age,
    "previous_year_rating": previous_year_rating,
    "length_of_service": length_of_service,
    "KPIs_met >80%": KPIs_met,
    "awards_won?": awards_won,
    "avg_training_score": avg_training_score
}])

# Convert to one-hot encoded format
input_encoded = pd.get_dummies(input_df)

# Match training columns
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Prediction
if st.button("Predict Promotion"):
    prediction = model.predict(input_encoded)

    if prediction[0] == 1:
        st.success(f"✅ Promotion Recommended using {selected_model_name}")
    else:
        st.error(f"❌ Not Recommended using {selected_model_name}")
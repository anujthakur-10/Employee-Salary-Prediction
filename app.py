import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict employee salary based on role, experience, education, and more.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_salary_dataset.csv")

df = load_data()

# Feature/Target split
X = df.drop("Salary(INR)", axis=1)
y = df["Salary(INR)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Columns
categorical_features = ["Position", "EducationLevel", "Industry", "Location"]
numerical_features = ["YearsExperience"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numerical_features)
])

# Full Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model.fit(X_train, y_train)

# --- Sidebar Inputs ---
st.sidebar.header("Enter Employee Details")
position = st.sidebar.selectbox("Position", sorted(df["Position"].unique()))
education = st.sidebar.selectbox("Education Level", sorted(df["EducationLevel"].unique()))
industry = st.sidebar.selectbox("Industry", sorted(df["Industry"].unique()))
location = st.sidebar.selectbox("Location", sorted(df["Location"].unique()))
experience = st.sidebar.slider("Years of Experience", min_value=0.0, max_value=40.0, step=0.5)

# Prediction function
def predict_salary(position, experience, education, industry, location, model):
    user_df = pd.DataFrame([{
        "Position": position,
        "YearsExperience": experience,
        "EducationLevel": education,
        "Industry": industry,
        "Location": location
    }])
    prediction = model.predict(user_df)[0]
    return round(prediction, 2)

# Predict Button
if st.button("🔍 Predict Salary"):
    salary = predict_salary(position, experience, education, industry, location, model)
    st.success(f"💰 Estimated Salary: ₹{salary:,.0f}")

# Model Evaluation Section
with st.expander("📊 Model Evaluation"):
    y_pred = model.predict(X_test)
    st.write(f"**MAE:** ₹{mean_absolute_error(y_test, y_pred):,.0f}")
    st.write(f"**RMSE:** ₹{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

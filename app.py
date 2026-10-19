
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Visit with Us | Purchase Prediction",
    page_icon="✈️",
    layout="centered"
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run the GitHub Actions pipeline first."
        )
    return joblib.load(MODEL_PATH)

st.title("✈️ Visit with Us")
st.subheader("Tourism Package Purchase Prediction")

st.write(
    "Enter customer information to estimate the likelihood "
    "of purchasing a tourism package before sales outreach."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

st.sidebar.header("About the Application")
st.sidebar.write(
    "This application uses a Random Forest model trained "
    "on historical tourism customer data."
)
st.sidebar.info(
    "Predictions support marketing decisions and do not "
    "guarantee a purchase."
)

with st.form("prediction_form"):

    st.header("Customer Details")

    age = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )

    city_tier = st.selectbox(
        "City Tier", [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Freelancer",
         "Large Business"]
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager",
         "AVP", "VP"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        max_value=1000000.0,
        value=30000.0,
        step=1000.0
    )

    st.header("Travel Preferences")

    number_of_trips = st.number_input(
        "Annual Number of Trips",
        min_value=0,
        max_value=30,
        value=3
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=15,
        value=0
    )

    preferred_property_star = st.selectbox(
        "Preferred Hotel Rating",
        [3, 4, 5]
    )

    passport = st.selectbox(
        "Valid Passport",
        ["No", "Yes"]
    )

    own_car = st.selectbox(
        "Owns a Car",
        ["No", "Yes"]
    )

    submitted = st.form_submit_button(
        "Predict Purchase Likelihood",
        use_container_width=True
    )

if submitted:

    if number_of_children_visiting > number_of_person_visiting:
        st.error(
            "Number of children cannot exceed total persons visiting."
        )
        st.stop()

    customer_data = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    # Align input columns with the fitted pipeline
    expected_columns = list(model.feature_names_in_)

    missing = set(expected_columns) - set(customer_data.columns)

    if missing:
        st.error(f"Missing required model features: {sorted(missing)}")
        st.stop()

    customer_data = customer_data[expected_columns]

    probability = float(
        model.predict_proba(customer_data)[0, 1]
    )

    prediction = int(probability >= 0.5)

    st.divider()
    st.header("Prediction Results")

    st.metric(
        "Estimated Purchase Probability",
        f"{probability * 100:.1f}%"
    )

    st.progress(probability)

    if prediction == 1:
        st.success(
            "Higher purchase propensity: consider this customer "
            "for marketing outreach."
        )
    else:
        st.info(
            "Lower purchase propensity: consider alternative "
            "campaigns or lower-priority outreach."
        )

    st.caption(
        "The 50% classification threshold is illustrative. "
        "It should be calibrated using marketing costs and "
        "campaign conversion outcomes."
    )

    with st.expander("View Model Input Data"):
        st.dataframe(customer_data, use_container_width=True)

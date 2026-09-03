import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="HEOR Copilot",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM DASHBOARD DESIGN
# ---------------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0, 188, 212, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(30, 136, 229, 0.18),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #071E2F 0%,
            #0B3045 50%,
            #082638 100%
        );
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0, 188, 212, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(30, 136, 229, 0.18),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #071E2F 0%,
            #0B3045 50%,
            #082638 100%
        );
}

/* Header */
[data-testid="stHeader"] {
    background: rgba(7, 30, 47, 0.85);
}

/* Main content */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* Main title */
h1 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

/* Section headings */
h2 {
    color: #7DE3F4 !important;
    font-weight: 700 !important;
}

h3 {
    color: #B8F3FA !important;
    font-weight: 600 !important;
}

/* Normal text */
p {
    color: #E6F7FA !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(125, 227, 244, 0.25);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
}

[data-testid="stMetricLabel"] {
    color: #A9DDE5 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Number inputs */
.stNumberInput input {
    background-color: rgba(255, 255, 255, 0.10) !important;
    color: white !important;
    border: 1px solid rgba(125, 227, 244, 0.35) !important;
    border-radius: 10px !important;
}

/* Select box */
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.10) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Prediction button */
div.stButton > button {
    background: linear-gradient(
        90deg,
        #00ACC1,
        #1976D2
    );
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(0, 172, 193, 0.3);
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #00C6D7,
        #2196F3
    );
    color: white !important;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #061A29;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏥 HEOR Copilot")

st.subheader(
    "Healthcare Economic & Outcomes Research"
)

st.write(
    "An interactive prototype for comparing healthcare "
    "treatments, evaluating costs and outcomes, and "
    "exploring hospitalization risk."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

data = pd.read_csv(
    "data/diabetes_heor_data.csv"
)


# ---------------------------------------------------------
# DERIVED HEOR VARIABLES
# ---------------------------------------------------------

data["HbA1c_Improvement"] = (
    data["HbA1c_Baseline"]
    - data["HbA1c_Followup"]
)

data["Hospitalization_Target"] = (
    data["Hospitalization"] == "Yes"
).astype(int)


# ---------------------------------------------------------
# LOAD ML MODEL
# ---------------------------------------------------------

ml_model = joblib.load(
    "RESULT/hospitalization_model.pkl"
)

st.success(
    f"Dataset loaded: {len(data):,} patients"
)


# ---------------------------------------------------------
# TREATMENT OVERVIEW
# ---------------------------------------------------------

st.header("📊 Treatment Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Treatment A Patients",
        int(
            (data["Treatment"] == "Treatment A").sum()
        )
    )

with col2:
    st.metric(
        "Treatment B Patients",
        int(
            (data["Treatment"] == "Treatment B").sum()
        )
    )


# ---------------------------------------------------------
# COST & EFFECTIVENESS
# ---------------------------------------------------------

st.header(
    "💰 Cost & Effectiveness Comparison"
)

comparison = data.groupby(
    "Treatment"
).agg(
    Average_Treatment_Cost=(
        "Treatment_Cost",
        "mean"
    ),
    Average_HbA1c_Improvement=(
        "HbA1c_Improvement",
        "mean"
    )
).round(2)

st.dataframe(
    comparison,
    use_container_width=True
)


# ---------------------------------------------------------
# HEALTHCARE OUTCOMES
# ---------------------------------------------------------

st.header(
    "🏥 Healthcare Outcomes"
)

outcomes = data.groupby(
    "Treatment"
).agg(
    Hospitalization_Rate=(
        "Hospitalization_Target",
        "mean"
    ),
    Average_Hospitalization_Cost=(
        "Hospitalization_Cost",
        "mean"
    )
)

outcomes["Hospitalization_Rate"] = (
    outcomes["Hospitalization_Rate"] * 100
).round(2)

outcomes["Total_Healthcare_Cost"] = (
    comparison["Average_Treatment_Cost"]
    + outcomes["Average_Hospitalization_Cost"]
)

outcomes = outcomes.round(2)

st.dataframe(
    outcomes,
    use_container_width=True
)


# ---------------------------------------------------------
# COST-EFFECTIVENESS / ICER
# ---------------------------------------------------------

st.header(
    "💰 Cost-Effectiveness Analysis"
)

cost_a = comparison.loc[
    "Treatment A",
    "Average_Treatment_Cost"
]

cost_b = comparison.loc[
    "Treatment B",
    "Average_Treatment_Cost"
]

effect_a = comparison.loc[
    "Treatment A",
    "Average_HbA1c_Improvement"
]

effect_b = comparison.loc[
    "Treatment B",
    "Average_HbA1c_Improvement"
]

incremental_cost = cost_b - cost_a

incremental_effect = effect_b - effect_a

icer = (
    incremental_cost
    / incremental_effect
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Incremental Cost",
        f"₹{incremental_cost:,.2f}"
    )

with col2:
    st.metric(
        "Additional HbA1c Improvement",
        f"{incremental_effect:.3f}"
    )

with col3:
    st.metric(
        "ICER",
        f"₹{icer:,.2f}"
    )

st.info(
    "ICER represents the additional treatment cost "
    "per additional unit of HbA1c improvement when "
    "choosing Treatment B over Treatment A."
)


# ---------------------------------------------------------
# TREATMENT COST CHART
# ---------------------------------------------------------

st.header(
    "📊 Treatment Cost Comparison"
)

st.bar_chart(
    comparison["Average_Treatment_Cost"]
)


# ---------------------------------------------------------
# HbA1c EFFECTIVENESS CHART
# ---------------------------------------------------------

st.header(
    "📉 HbA1c Effectiveness Comparison"
)

st.bar_chart(
    comparison["Average_HbA1c_Improvement"]
)


# ---------------------------------------------------------
# ML HOSPITALIZATION PREDICTION
# ---------------------------------------------------------

st.header(
    "🤖 Hospitalization Risk Prediction"
)

st.write(
    "Enter patient characteristics to estimate "
    "hospitalization risk."
)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=50
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    comorbidities = st.number_input(
        "Number of Comorbidities",
        min_value=0,
        max_value=10,
        value=1
    )


with col2:

    treatment_duration = st.number_input(
        "Treatment Duration (Months)",
        min_value=1,
        max_value=60,
        value=12
    )

    hba1c_baseline = st.number_input(
        "Baseline HbA1c",
        min_value=4.0,
        max_value=15.0,
        value=7.0
    )

    treatment = st.selectbox(
        "Treatment",
        [
            "Treatment A",
            "Treatment B"
        ]
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button(
    "Predict Hospitalization Risk"
):

    patient = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Comorbidities": [comorbidities],
        "Treatment_Duration_Months": [
            treatment_duration
        ],
        "HbA1c_Baseline": [
            hba1c_baseline
        ]
    })

    prediction = ml_model.predict(
        patient
    )

    probability = ml_model.predict_proba(
        patient
    )[0][1]

    st.subheader(
        "Prediction Result"
    )

    st.metric(
        "Estimated Hospitalization Risk",
        f"{probability:.1%}"
    )

    if prediction[0] == 1:

        st.warning(
            "The model predicts a higher likelihood "
            "of hospitalization."
        )

    else:

        st.success(
            "The model predicts a lower likelihood "
            "of hospitalization."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.caption(
    "This is an educational HEOR prototype using "
    "simulated data and is not intended for clinical "
    "decision-making."
)
import streamlit as st
import pickle
import pandas as pd

# -------------------------
# Load Model
# -------------------------
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

treatments = ["Chemotherapy", "Hormonal Therapy", "Surgery"]

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Breast Cancer AI System",
    layout="wide"
)

# -------------------------
# Header
# -------------------------
st.title(" Breast Cancer Treatment Recommendation System")
st.markdown("AI-powered clinical decision support system")
st.markdown("---")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Patient Clinical Data")

age = st.sidebar.slider("Age", 20, 90, 50)

menopausal = st.sidebar.selectbox("Menopausal Status", ["Pre", "Post"])
er = st.sidebar.selectbox("ER Status", ["Positive", "Negative"])
pr = st.sidebar.selectbox("PR Status", ["Positive", "Negative"])
her2 = st.sidebar.selectbox("HER2 Status", ["Positive", "Negative"])

stage = st.sidebar.selectbox("Cancer Stage", ["I", "II", "III", "IV"])
t_status = st.sidebar.selectbox("T Status", ["T1", "T2", "T3", "T4"])
n_status = st.sidebar.selectbox("N Status", ["N0", "N1", "N2", "N3"])
m_status = st.sidebar.selectbox("M Status", ["M0", "M1"])

lymph = st.sidebar.slider("Lymph Node Count", 0, 30, 5)

necrosis = st.sidebar.selectbox("Tumor Necrosis", ["Low", "Medium", "High"])
histology = st.sidebar.selectbox("Histology", ["Ductal", "Lobular", "Mixed"])
location = st.sidebar.selectbox(
    "Tumor Location",
    ["Upper Outer", "Upper Inner", "Lower Outer", "Lower Inner"]
)

# -------------------------
# Encode Function
# -------------------------
def encode(x):
    return hash(x) % 3

# -------------------------
# FIXED INPUT DATA (correct format)
# -------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Menopausal_Status": encode(menopausal),
    "ER_Status": encode(er),
    "PR_Status": encode(pr),
    "HER2_Status": encode(her2),
    "Cancer_Stage": encode(stage),
    "T_Status": encode(t_status),
    "N_Status": encode(n_status),
    "M_Status": encode(m_status),
    "Lymph_Node_Count": lymph,
    "Tumor_Necrosis": encode(necrosis),
    "Histology_Type": encode(histology),
    "Tumor_Location": encode(location)
}])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Treatment"):

    # IMPORTANT FIX: DO NOT WRAP AGAIN
    input_df = input_data

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.markdown("##  Prediction Result")

    st.success(f"Recommended Treatment: **{treatments[pred]}**")

    confidence = prob[pred] * 100
    st.info(f"Confidence: {confidence:.2f}%")

    st.markdown("###  Probability Breakdown")

    col1, col2, col3 = st.columns(3)

    for i, (t, p) in enumerate(zip(treatments, prob)):
        with [col1, col2, col3][i]:
            st.metric(label=t, value=f"{p*100:.2f}%")
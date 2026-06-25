import pandas as pd
import pickle

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Treatment names
treatments = [
    "Chemotherapy",
    "Hormonal Therapy",
    "Surgery"
]

# --------------------------------
# Test Patient
# --------------------------------

patient = pd.DataFrame([{
    "Age": 60,
    "Menopausal_Status": 1,
    "ER_Status": 1,
    "PR_Status": 1,
    "HER2_Status": 0,
    "Cancer_Stage": 1,
    "T_Status": 1,
    "N_Status": 0,
    "M_Status": 0,
    "Lymph_Node_Count": 2,
    "Tumor_Necrosis": 1,
    "Histology_Type": 0,
    "Tumor_Location": 0
}])

prediction = model.predict(patient)

probabilities = model.predict_proba(patient)[0]

print("\nPredicted Class:")
print(prediction[0])

print("\nProbabilities:")

for treatment, prob in zip(treatments, probabilities):
    print(f"{treatment}: {prob*100:.2f}%")
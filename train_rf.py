import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -----------------------------
# Load Dataset


df = pd.read_csv(
    "dataset/breast_cancer_treatment_dataset_v3_clean.csv"
)

print("\nDataset Shape:")
print(df.shape)

# -----------------------------
# Encode Categorical Columns


encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

# CHECK TREATMENT ENCODING
print("\nTreatment Encoding:")
print(encoders["Treatment"].classes_)

#-------------------------------
# Features & Target


X = df.drop("Treatment", axis=1)

y = df["Treatment"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Random Forest
# -----------------------------

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

y_pred = rf.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# -----------------------------
# Save Model
# -----------------------------

with open(
    "random_forest_model.pkl",
    "wb"
) as file:

    pickle.dump(
        rf,
        file
    )

print("\nModel Saved Successfully")

# Save Encoders

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Encoders Saved Successfully")
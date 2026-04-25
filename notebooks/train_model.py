import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Loads the dataset
df = pd.read_csv("data/patients_dakar.csv")

# Verify the dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} columns")
print(f"\nColumns : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")



# Encode categorical variables as numbers
# The model only includes numbers
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

print(f"="*60)

# Define the features (X) and the target (y)
features_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_encoded']

X = df[features_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Labels : {y.shape}")

print(f"="*60)

# 80% for the training, 20% for the test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,    # 20% for the test
    random_state = 42,  # To have the same results each time
    stratify = y        # Keep the same diagnostic proportions
)

print(f"Training : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

print(f"="*60)

# Create the model
model = RandomForestClassifier(
    n_estimators = 100, # 100 decision trees
    random_state = 42,  # Reproducibility
)

# Train on the training data
model.fit(X_train, y_train)

print("Model trained")
print(f"Number of trees : {model.n_estimators}")
print(f"Number of features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

print(f"="*60)

# Predicting based on test data
y_pred = model.predict(X_test)

# Compare the first 10 predictions with reality
comparison = pd.DataFrame({
    'True diagnosis': y_test.values[:10],
    'Prediction': y_pred[:10]
})

print(comparison)

print(f"="*60)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

print(f"="*60)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Confusion matrix:")
print(cm)

# Classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print(f"="*60)

# Visualize with seaborn
plt.figure(figsize = (8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Model prediction")
plt.ylabel("True diagnosis")
plt.title("Confusion matrix - SenSante")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.show()

print("Figure saved as: figures/confusion_matrix.png")

print("="*60)

# Create the models/ folder if it does not exist
os.makedirs("models", exist_ok = True)

# Serialize the model
joblib.dump(model, "models/model.pkl")

# Check the file size
size = os.path.getsize("models/model.pkl")
print(f"Model saved as : models/model.pkl")
print(f"Size : {size / 1024:.1f} Ko")

print("="*60)

# Save the encoders (essential for the new data)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Save the list of features (for reference)
joblib.dump(features_cols, "models/features_cols.pkl")

print("Encoders & Features saved.")

print("")
print("–"*100)
print("")

# Simulate what the API will do in Lab 3:
# Load the model FROM THE FILE (not from memory)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Model loaded : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# A new patient arrives at the Medina health center.
new_patient = {
    'age'        : 28,
    'sexe'       : 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux'       : True,
    'fatigue'    : True,
    'maux_tete'  : True,
    'frissons'   : True,
    'nausee'     : True,
    'region'     : 'Dakar'
}

# Encode category values
sexe_enc = le_sexe_loaded.transform([new_patient['sexe']])[0]
region_enc = le_region_loaded.transform([new_patient['region']])[0]

# Prepare the feature vector
features = [
    new_patient['age'],
    sexe_enc,
    new_patient['temperature'],
    new_patient['tension_sys'],
    int(new_patient['toux']),
    int(new_patient['fatigue']),
    int(new_patient['maux_tete']),
    int(new_patient['frissons']),
    int(new_patient['nausee']),
    region_enc
]

# Predict
#diagnostic = model_loaded.predict([features])[0]
#probas = model_loaded.predict_proba([features])[0]
features_df = pd.DataFrame([features], columns=features_cols)
diagnostic = model_loaded.predict(features_df)[0]
probas = model_loaded.predict_proba(features_df)[0]
proba_max = probas.max()

print(f"\n --- Pre-diagnosis result ---")
print(f"Patient : {new_patient['sexe']}, {new_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probability : {proba_max:.1f} %")
print(f"\nProbability per class :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%} {bar}")

print("")
print("–"*100)
print("")

# Exercise
importance = model.feature_importances_
for name, imp in sorted(zip(features_cols,  importance), key=lambda  x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3%}")

print("=" * 60)

# 3 patients fictifs
patients = [
    {
        "nom": "Patient 1 - Jeune sans symptômes",
        "age": 22, "sexe": "M", "temperature": 36.8,
        "tension_sys": 120, "toux": False, "fatigue": False,
        "maux_tete": False, "frissons": False, "nausee": False,
        "region": "Dakar"
    },
    {
        "nom": "Patient 2 - Adulte avec forte fièvre",
        "age": 35, "sexe": "F", "temperature": 40.1,
        "tension_sys": 100, "toux": True, "fatigue": True,
        "maux_tete": True, "frissons": True, "nausee": False,
        "region": "Thiès"
    },
    {
        "nom": "Patient 3 - Patient âgé avec toux",
        "age": 65, "sexe": "M", "temperature": 38.2,
        "tension_sys": 135, "toux": True, "fatigue": True,
        "maux_tete": False, "frissons": False, "nausee": True,
        "region": "Saint-Louis"
    }
]

for patient in patients:
    sexe_enc = le_sexe_loaded.transform([patient["sexe"]])[0]
    region_enc = le_region_loaded.transform([patient["region"]])[0]

    patient_features = [
        patient["age"], sexe_enc, patient["temperature"],
        patient["tension_sys"], int(patient["toux"]),
        int(patient["fatigue"]), int(patient["maux_tete"]),
        int(patient["frissons"]), int(patient["nausee"]),
        region_enc
    ]

    features_df = pd.DataFrame([patient_features], columns=features_cols)
    diagnostic = model_loaded.predict(features_df)[0]
    probas = model_loaded.predict_proba(features_df)[0]

    print(f"\n{'='*45}")
    print(f" {patient['nom']}")
    print(f" {patient['sexe']}, {patient['age']} ans, T={patient['temperature']}°C")
    print(f" Diagnostic : {diagnostic} ({probas.max():.1%})")
    print(f" Probabilités :")
    for classe, proba in zip(model_loaded.classes_, probas):
        bar = '#' * int(proba * 30)
        print(f"   {classe:8s} : {proba:6.1%} {bar}")
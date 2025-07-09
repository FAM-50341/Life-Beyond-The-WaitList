# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
import dask.dataframe as dd
import time
import warnings
warnings.filterwarnings('ignore')

# Start time tracking
start_time = time.time()

# Google Drive mount
drive.mount('/content/drive')

# Dataset load with Dask
file_path = '/content/drive/My Drive/TransplantData/matches.csv'
df = dd.from_pandas(pd.read_csv(file_path), npartitions=4)
patients = pd.read_csv('/content/drive/My Drive/TransplantData/patients.csv')
donors = pd.read_csv('/content/drive/My Drive/TransplantData/donors.csv')

# Convert to pandas
df = df.compute()

# Check for missing values
print("Missing values in df:\n", df.isnull().sum())
df = df.dropna()
print(f"Dataset size after dropping NA: {df.shape}")

# Verify target column
target_col = 'Transplantation Eligibility'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

# Store LabelEncoders
label_encoders = {}

# Age and Gender merge
df = df.merge(patients[['Patient ID', 'Age', 'Gender']], on='Patient ID', how='inner')
df = df.merge(donors[['Donor ID', 'Age', 'Gender']], on='Donor ID', how='inner', suffixes=('_Patient', '_Donor'))

# Split Blood Group
df[['Donor Blood Group', 'Patient Blood Group']] = df['Blood Group Match'].str.split(' to ', expand=True)
df = df.drop(columns=['Blood Group Match'])

# Drop unnecessary columns
df = df.drop(columns=['Patient ID', 'Donor ID'])

# Feature engineering
df['Age_Difference'] = abs(df['Age_Patient'] - df['Age_Donor'])

# Encode Gender
df['Gender_Patient'] = df['Gender_Patient'].map({'Male': 0, 'Female': 1})
df['Gender_Donor'] = df['Gender_Donor'].map({'Male': 0, 'Female': 1})

# Encode categorical columns
categorical_cols = ['Organ', 'HLA Match', 'Location', 'Donor Blood Group', 'Patient Blood Group']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
df[target_col] = le_target.fit_transform(df[target_col])
label_encoders[target_col] = le_target

# Define features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)

# Print class distribution
print("Class distribution before resampling:\n", y_train.value_counts(normalize=True))
print("\nClass distribution after resampling:\n", pd.Series(y_train_res).value_counts(normalize=True))

# Define individual models
xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=8, scale_pos_weight=2)
dt_model = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=10, class_weight='balanced')
lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=50, max_depth=8, num_leaves=31, is_unbalance=True)
rf_model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=8, class_weight='balanced')

# Reduced tuning
lgb_params = {'n_estimators': [50, 100], 'max_depth': [5, 8], 'num_leaves': [15, 31]}
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 8]}
lgb_params = RandomizedSearchCV(lgb_model, lgb_params, n_iter=4, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=4, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)

lgb_search.fit(X_train_res, y_train_res)
rf_search.fit(X_train_res, y_train_res)

lgb_model = lgb_search.best_estimator_
rf_model = rf_search.best_estimator_
print("Best LightGBM params:", lgb_search.best_params_)
print("Best Random Forest params:", rf_search.best_params_)

# Feature importance
lgb_model.fit(X_train_res, y_train_res)
importances = pd.DataFrame({'feature': X.columns, 'importance': lgb_model.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print("\nTop 5 features:\n", importances.head())

# Select top features
top_features = importances[importances['importance'] > 0]['feature'].tolist()
X_train_res = X_train_res[top_features]
X_test = X_test[top_features]

# Create stacking ensemble
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb_model), ('dt', dt_model), ('lgb', lgb_model), ('rf', rf_model)],
    final_estimator=LogisticRegression(),
    cv=None,
    n_jobs=-1
)

# Train stacking ensemble
stacking_model.fit(X_train_res, y_train_res)

# Evaluate on test set
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# User input function
def get_user_input():
    print("\nEnter patient and donor information:")
    try:
        patient_id = input("Patient ID (e.g., P123): ")
        patient_age = float(input("Patient Age (e.g., 45): "))
        patient_gender = input("Patient Gender (Male/Female): ").capitalize()
        patient_blood_group = input("Patient Blood Group (e.g., A+): ")
        patient_location = input("Patient Location (e.g., New York): ")

        donor_id = input("Donor ID (e.g., D123): ")
        donor_age = float(input("Donor Age (e.g., 40): "))
        donor_gender = input("Donor Gender (Male/Female): ").capitalize()
        donor_blood_group = input("Donor Blood Group (e.g., A+): ")

        organ = input("Organ (e.g., Kidney, Liver): ").capitalize()
        hla_match = input("HLA Match (e.g., A*01:01): ")
        location = input("Transplant Location (e.g., New York): ")

        # Validate inputs
        if patient_gender not in ['Male', 'Female'] or donor_gender not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'.")
        if patient_age < 0 or donor_age < 0:
            raise ValueError("Age cannot be negative.")
        if not patient_blood_group or not donor_blood_group:
            raise ValueError("Blood groups cannot be empty.")

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age_Patient': [patient_age],
            'Gender_Patient': [patient_gender],
            'Patient Blood Group': [patient_blood_group],
            'Location_Patient': [patient_location],
            'Age_Donor': [donor_age],
            'Gender_Donor': [donor_gender],
            'Donor Blood Group': [donor_blood_group],
            'Organ': [organ],
            'HLA Match': [hla_match],
            'Location': [location]
        })
        return input_data
    except Exception as e:
        print(f"Error in input: {e}")
        return None

# Preprocess user input
def preprocess_input(input_data, label_encoders, top_features):
    try:
        # Feature engineering
        input_data['Age_Difference'] = abs(input_data['Age_Patient'] - input_data['Age_Donor'])

        # Encode Gender
        input_data['Gender_Patient'] = input_data['Gender_Patient'].map({'Male': 0, 'Female': 1})
        input_data['Gender_Donor'] = input_data['Gender_Donor'].map({'Male': 0, 'Female': 1})

        # Debug: Print input before encoding
        print("\nInput before encoding:\n", input_data)

        # Encode categorical columns
        for col in ['Organ', 'HLA Match', 'Location', 'Donor Blood Group', 'Patient Blood Group']:
            if col in input_data.columns:
                le = label_encoders.get(col)
                if le:
                    # Handle unseen categories
                    input_data[col] = input_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    input_data[col] = le.transform(input_data[col])
                else:
                    raise ValueError(f"No LabelEncoder found for {col}")

        # Rename Location_Patient if necessary (not used in model)
        if 'Location_Patient' in input_data.columns:
            input_data = input_data.drop(columns=['Location_Patient'])

        # Debug: Print input after encoding
        print("\nInput after encoding:\n", input_data)
        print("\nDtypes after encoding:\n", input_data.dtypes)

        # Select top features
        missing_cols = [col for col in top_features if col not in input_data.columns]
        for col in missing_cols:
            input_data[col] = 0  # Fill missing with 0
        input_data = input_data[top_features]

        # Ensure all columns are numerical
        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                raise ValueError(f"Column {col} is still object type: {input_data[col].values}")

        return input_data
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

# Predict for user input
while True:
    input_data = get_user_input()
    if input_data is None:
        print("Invalid input. Try again? (y/n)")
        if input("Enter y/n: ").lower() != 'y':
            break
        continue

    processed_input = preprocess_input(input_data, label_encoders, top_features)
    if processed_input is None:
        print("Failed to process input. Try again? (y/n)")
        if input("Enter y/n: ").lower() != 'y':
            break
        continue

    # Predict
    prediction = stacking_model.predict(processed_input)
    probability = stacking_model.predict_proba(processed_input)[0][1]
    result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
    print(f"\nPrediction: {result}")
    print(f"Probability of Eligibility: {probability:.2%}")

    # Ask to continue
    if input("\nPredict another? (y/n): ").lower() != 'y':
        break

# Total time
print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

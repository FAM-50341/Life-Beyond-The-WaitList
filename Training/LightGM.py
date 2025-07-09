# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
import time

# Start time tracking
start_time = time.time()

# Google Drive mount
drive.mount('/content/drive')

# Dataset load
file_path = '/content/drive/My Drive/TransplantData/matches.csv'
df = pd.read_csv(file_path)

patients = pd.read_csv('/content/drive/My Drive/TransplantData/patients.csv')
donors = pd.read_csv('/content/drive/My Drive/TransplantData/donors.csv')

# Check for missing values
print("Missing values in df:\n", df.isnull().sum())
print("Missing values in patients:\n", patients.isnull().sum())
print("Missing values in donors:\n", donors.isnull().sum())
df = df.dropna()  # Drop missing values if any
print(f"Dataset size after dropping NA: {df.shape}")

# Columns check
print("Columns in df:", df.columns.tolist())
print("Columns in patients:", patients.columns.tolist())
print("Columns in donors:", donors.columns.tolist())
print(f"Original dataset size: {df.shape}")

# Verify target column
target_col = 'Transplantation Eligibility'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
print(f"Using target column: {target_col}")

# Age and Gender merge
df = df.merge(patients[['Patient ID', 'Age', 'Gender']], on='Patient ID', how='left')
print(f"Columns after patient merge: {df.columns.tolist()}")
df = df.merge(donors[['Donor ID', 'Age', 'Gender']], on='Donor ID', how='left', suffixes=('_Patient', '_Donor'))
print(f"Columns after donor merge: {df.columns.tolist()}")

# Split Blood Group Match
df[['Donor Blood Group', 'Patient Blood Group']] = df['Blood Group Match'].str.split(' to ', expand=True)
df = df.drop('Blood Group Match', axis=1)
print(f"Columns after splitting Blood Group Match: {df.columns.tolist()}")

# Drop unnecessary columns
df = df.drop(['Patient ID', 'Donor ID'], axis=1)
print(f"Columns after dropping Patient ID, Donor ID: {df.columns.tolist()}")

# Encode Gender (Male = 0, Female = 1)
df['Gender_Patient'] = df['Gender_Patient'].map({'Male': 0, 'Female': 1})
df['Gender_Donor'] = df['Gender_Donor'].map({'Male': 0, 'Female': 1})

# Encode categorical columns with integers (for LightGBM native support)
categorical_cols = ['Organ', 'HLA Match', 'Location', 'Donor Blood Group', 'Patient Blood Group']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
print(f"Columns after integer encoding: {df.columns.tolist()}")

# Encode target variable
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
print(f"Columns after encoding target: {df.columns.tolist()}")

# Define features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Print class distribution
print("Class distribution before resampling:\n", y_train.value_counts())
print("\nClass distribution after resampling:\n", pd.Series(y_train_resampled).value_counts())

# LightGBM model with categorical features
lgb_model = lgb.LGBMClassifier(random_state=42, is_unbalance=True, n_estimators=100, max_depth=10, num_leaves=31)
lgb_model.fit(X_train_resampled, y_train_resampled, categorical_feature=categorical_cols)

# Prediction
y_pred = lgb_model.predict(X_test)

# Accuracy and classification report
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_prob = lgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Total time
print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

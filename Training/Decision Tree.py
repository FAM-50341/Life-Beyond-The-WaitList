# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
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

# Columns check
print("Columns in df:", df.columns.tolist())
print("Columns in patients:", patients.columns.tolist())
print("Columns in donors:", donors.columns.tolist())
print(f"Original dataset size: {df.shape}")

# Age and Gender merge
df = df.merge(patients[['Patient ID', 'Age', 'Gender']], on='Patient ID', how='left')
df = df.merge(donors[['Donor ID', 'Age', 'Gender']], on='Donor ID', how='left', suffixes=('_Patient', '_Donor'))

# Split Blood Group Match
df[['Donor Blood Group', 'Patient Blood Group']] = df['Blood Group Match'].str.split(' to ', expand=True)
df = df.drop('Blood Group Match', axis=1)

# Drop unnecessary columns
df = df.drop(['Patient ID', 'Donor ID'], axis=1)

# Encode Gender (Male = 0, Female = 1)
df['Gender_Patient'] = df['Gender_Patient'].map({'Male': 0, 'Female': 1})
df['Gender_Donor'] = df['Gender_Donor'].map({'Male': 0, 'Female': 1})

# Encode target variable
le = LabelEncoder()
df['Transplantation Eligibility'] = le.fit_transform(df['Transplantation Eligibility'])

# One-hot encode categorical columns
categorical_cols = ['Organ', 'HLA Match', 'Location', 'Donor Blood Group', 'Patient Blood Group']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target (no scaling needed for Decision Tree)
X = df.drop('Transplantation Eligibility', axis=1)
y = df['Transplantation Eligibility']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use RandomUnderSampler for class imbalance
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Print class distribution
print("Class distribution before resampling:\n", y_train.value_counts())
print("\nClass distribution after resampling:\n", pd.Series(y_train_resampled).value_counts())

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10, min_samples_split=10)
dt_model.fit(X_train_resampled, y_train_resampled)

# Prediction
y_pred = dt_model.predict(X_test)

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
y_prob = dt_model.predict_proba(X_test)[:, 1]  # Use predict_proba for Decision Tree
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

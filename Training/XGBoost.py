import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.preprocessing import LabelEncoder

# Mount Google Drive
drive.mount('/content/drive')

# Load the datasets
file_path = '/content/drive/My Drive/TransplantData/matches.csv'
df = pd.read_csv(file_path)

patients = pd.read_csv('/content/drive/My Drive/TransplantData/patients.csv')
donors = pd.read_csv('/content/drive/My Drive/TransplantData/donors.csv')

# Check columns to ensure 'Patient ID' and 'Donor ID' exist
print("Columns in df:", df.columns.tolist())
print("Columns in patients:", patients.columns.tolist())
print("Columns in donors:", donors.columns.tolist())

# Merge Age and Gender from patients and donors
df = df.merge(patients[['Patient ID', 'Age', 'Gender']], on='Patient ID', how='left')
df = df.merge(donors[['Donor ID', 'Age', 'Gender']], on='Donor ID', how='left', suffixes=('_Patient', '_Donor'))

# Split 'Blood Group Match' into 'Donor Blood Group' and 'Patient Blood Group'
df[['Donor Blood Group', 'Patient Blood Group']] = df['Blood Group Match'].str.split(' to ', expand=True)
df = df.drop('Blood Group Match', axis=1)

# Drop unnecessary columns
df = df.drop(['Patient ID', 'Donor ID'], axis=1)

# Encode the target variable 'Transplantation Eligibility' (assuming 'Yes'/'No')
le = LabelEncoder()
df['Transplantation Eligibility'] = le.fit_transform(df['Transplantation Eligibility'])

# Encode 'Gender_Patient' and 'Gender_Donor' (assuming 'Male'/'Female')
df['Gender_Patient'] = df['Gender_Patient'].map({'Male': 0, 'Female': 1})
df['Gender_Donor'] = df['Gender_Donor'].map({'Male': 0, 'Female': 1})

# Apply one-hot encoding to categorical columns
categorical_cols = ['Organ', 'HLA Match', 'Location', 'Donor Blood Group', 'Patient Blood Group']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
X = df.drop('Transplantation Eligibility', axis=1)
y = df['Transplantation Eligibility']

# Convert all columns in X to integers (post one-hot encoding, columns are uint8, but ensuring int type)
X = X.astype(int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy=0.1, random_state=42)  # 10% "No" samples relative to "Yes"
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print class distribution
print("Class distribution before SMOTE:\n", y_train.value_counts())
print("\nClass distribution after SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# Train XGBoost with hyperparameter tuning
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("\nScale Pos Weight:", scale_pos_weight)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3]
}
model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model
model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print accuracy and classification report
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Print and visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Threshold tuning
threshold = 0.4  # Adjust to increase "No" predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred_threshold = (y_prob >= threshold).astype(int)
print("\nAccuracy with Threshold 0.4:", accuracy_score(y_test, y_pred_threshold))
print("\nClassification Report with Threshold 0.4:\n", classification_report(y_test, y_pred_threshold, target_names=['No', 'Yes']))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Print feature importance
print("\nFeature Importance:\n", feature_importance)

# ROC Curve
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

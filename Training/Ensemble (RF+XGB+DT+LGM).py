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

# Convert back to pandas after loading
df = df.compute()

# Check for missing values
print("Missing values in df:\n", df.isnull().sum())
df = df.dropna()
print(f"Dataset size after dropping NA: {df.shape}")

# Verify target column
target_col = 'Transplantation Eligibility'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

# Age and Gender merge
df = df.merge(patients[['Patient ID', 'Age', 'Gender']], on='Patient ID', how='inner')
df = df.merge(donors[['Donor ID', 'Age', 'Gender']], on='Donor ID', how='inner', suffixes=('_Patient', '_Donor'))

# Split Blood Group Match
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
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode target variable
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

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

# Define individual models with fewer estimators
xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=8, scale_pos_weight=2)
dt_model = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=10, class_weight='balanced')
lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=50, max_depth=8, num_leaves=31, is_unbalance=True)
rf_model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=8, class_weight='balanced')

# Reduced tuning for LightGBM and Random Forest
lgb_params = {'n_estimators': [50, 100], 'max_depth': [5, 8], 'num_leaves': [15, 31]}
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 8]}
lgb_search = RandomizedSearchCV(lgb_model, lgb_params, n_iter=4, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
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

# Create stacking ensemble without cross-validation
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb_model), ('dt', dt_model), ('lgb', lgb_model), ('rf', rf_model)],
    final_estimator=LogisticRegression(),
    cv=None,
    n_jobs=-1
)

# Train stacking ensemble
stacking_model.fit(X_train_res, y_train_res)

# Prediction
y_pred = stacking_model.predict(X_test)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
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
y_prob = stacking_model.predict_proba(X_test)[:, 1]
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

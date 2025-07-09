import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load preprocessed dataset
file_path = '/content/drive/My Drive/TransplantData/matches_preprocessed.csv'
df = pd.read_csv(file_path)

# Step 1: Prepare features and target
X = df.drop('Transplantation Eligibility', axis=1)
y = df['Transplantation Eligibility']

# Convert boolean columns to integers
X = X.astype(int)

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print class distribution before and after SMOTE
print("Class distribution before SMOTE:\n", y_train.value_counts())
print("\nClass distribution after SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# Step 4: Train RandomForestClassifier
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X_train_resampled, y_train_resampled)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)

# Print accuracy and classification report
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 6: Feature Importance
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

# Step 7: Save the model
model_path = '/content/drive/My Drive/TransplantData/transplant_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

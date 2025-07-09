import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to matches.csv
file_path = '/content/drive/My Drive/TransplantData/matches.csv'  # Adjust path as needed

# Load the dataset
df = pd.read_csv(file_path)

# Step 1: Check for missing values
print("Missing values:\n", df.isnull().sum())

# Step 2: Drop unnecessary columns
df = df.drop(['Patient ID', 'Donor ID'], axis=1)

# Step 3: Split 'Blood Group Match' into donor and patient blood groups
df[['Donor Blood Group', 'Patient Blood Group']] = df['Blood Group Match'].str.split(' to ', expand=True)
df = df.drop('Blood Group Match', axis=1)

# Step 4: Encode categorical variables
# Label encode the target variable
le = LabelEncoder()
df['Transplantation Eligibility'] = le.fit_transform(df['Transplantation Eligibility'])  # Yes=1, No=0

# One-hot encode categorical features
categorical_cols = ['Organ', 'HLA Match', 'Location', 'Donor Blood Group', 'Patient Blood Group']
df = pd.get_dummies(df, columns=categorical_cols)

# Step 5: Check class balance
print("Class distribution:\n", df['Transplantation Eligibility'].value_counts())

# Step 6: Save preprocessed dataset
output_folder = '/content/drive/My Drive/TransplantData/'
os.makedirs(output_folder, exist_ok=True)
df.to_csv(f'{output_folder}matches_preprocessed.csv', index=False)
print(f"Preprocessed dataset saved to {output_folder}matches_preprocessed.csv")

# Display the first few rows of the preprocessed data
print("\nPreprocessed data sample:\n", df.head())

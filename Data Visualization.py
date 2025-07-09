import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.preprocessing import LabelEncoder

# Mount Google Drive
drive.mount('/content/drive')

# Load the original matches.csv
file_path = '/content/drive/My Drive/TransplantData/matches.csv'
df = pd.read_csv(file_path)

# Step 1: Summary Statistics
print("Dataset Shape:", df.shape)
print("\nSummary Statistics for Categorical Features:")
print(df.describe(include='object'))

# Step 2: Split Blood Group Match for analysis
df[['Donor Blood Group', 'Patient Blood Group']] = df['Blood Group Match'].str.split(' to ', expand=True)

# Step 3: Print value counts for key features
print("\nTransplantation Eligibility Counts:\n", df['Transplantation Eligibility'].value_counts())
print("\nOrgan Counts:\n", df['Organ'].value_counts())
print("\nHLA Match Counts:\n", df['HLA Match'].value_counts())
print("\nLocation Counts:\n", df['Location'].value_counts())
print("\nDonor Blood Group Counts:\n", df['Donor Blood Group'].value_counts())
print("\nPatient Blood Group Counts:\n", df['Patient Blood Group'].value_counts())

# Step 4: Visualizations
# Set plot style
sns.set_style("whitegrid")

# Chart 1: Transplantation Eligibility Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Transplantation Eligibility', data=df, palette=['#36A2EB', '#FF6384'])
plt.title('Distribution of Transplantation Eligibility')
plt.xlabel('Eligibility')
plt.ylabel('Count')
plt.show()

# Chart 2: Organ Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Organ', data=df, palette=['#4BC0C0', '#FFCE56', '#FF9F40'], order=['Kidney', 'Liver', 'Heart'])
plt.title('Distribution of Organ Types')
plt.xlabel('Organ Type')
plt.ylabel('Count')
plt.show()

# Chart 3: Stacked Bar Chart for Eligibility by Organ
eligibility_by_organ = df.groupby(['Organ', 'Transplantation Eligibility']).size().unstack(fill_value=0)
plt.figure(figsize=(8, 6))
eligibility_by_organ.plot(kind='bar', stacked=True, color=['#36A2EB', '#FF6384'])
plt.title('Transplantation Eligibility by Organ')
plt.xlabel('Organ Type')
plt.ylabel('Count')
plt.legend(title='Eligibility')
plt.show()

# Chart 4: Location Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Location', data=df, palette=['#9966FF', '#FF6684', '#66FF99', '#FFCC66', '#66CCCC', '#CC99FF'],
              order=['Dhaka', 'Chattogram', 'Rajshahi', 'Sylhet', 'Khulna', 'Barisal'])
plt.title('Distribution of Locations')
plt.xlabel('Location')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Step 5: Correlation Analysis
# Select only columns to encode
df_encoded = df[['Organ', 'HLA Match', 'Location', 'Donor Blood Group', 'Patient Blood Group', 'Transplantation Eligibility']].copy()
le = LabelEncoder()
for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])
print("\nCorrelation Matrix:\n", df_encoded.corr())

# Compute correlation matrix
corr_matrix = df_encoded.corr()

# Chart 5: Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
plt.title('Correlation Matrix of Encoded Features')
plt.show()

# Print correlation matrix for reference
print("\nCorrelation Matrix:\n", corr_matrix)

import random
import csv
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the folder path in Google Drive where CSVs will be saved
output_folder = '/content/drive/My Drive/TransplantData/'  # Change this path as needed

# Data distributions (unchanged)
blood_groups = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
blood_weights = [0.4, 0.07, 0.34, 0.06, 0.1, 0.02, 0.01, 0.01]
organs = ['Kidney', 'Liver', 'Heart']
organ_weights = [0.5, 0.3, 0.2]
conditions = ['Kidney Failure', 'Chronic Renal Disease', 'Acute Renal Failure',
              'Cirrhosis', 'Liver Failure', 'Cardiomyopathy', 'Heart Disease']
locations = ['Dhaka', 'Chattogram', 'Rajshahi', 'Sylhet', 'Khulna', 'Barisal']
location_weights = [0.4, 0.3, 0.15, 0.1, 0.03, 0.02]
hla_types = ['HLA-A,B,DR', 'HLA-A,C,DR', 'HLA-A,B,C']
health_status = ['Healthy', 'Deceased']
health_weights = [0.8, 0.2]
genders = ['Male', 'Female']

# Blood group compatibility (unchanged)
blood_compatibility = {
    'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],
    'O+': ['O+', 'A+', 'B+', 'AB+'],
    'A-': ['A-', 'A+', 'AB-', 'AB+'],
    'A+': ['A+', 'AB+'],
    'B-': ['B-', 'B+', 'AB-', 'AB+'],
    'B+': ['B+', 'AB+'],
    'AB-': ['AB-', 'AB+'],
    'AB+': ['AB+']
}

# Generate patients (unchanged)
patients = []
for i in range(5000):
    patient = {
        'Patient ID': f'P{i+1}',
        'Age': random.randint(18, 70),
        'Gender': random.choice(genders),
        'Blood Group': random.choices(blood_groups, blood_weights)[0],
        'Required Organ': random.choices(organs, organ_weights)[0],
        'Condition': random.choice(conditions),
        'Location': random.choices(locations, location_weights)[0],
        'HLA Typing': random.choice(hla_types)
    }
    patients.append(patient)

# Generate donors (unchanged)
donors = []
for i in range(5000):
    donor = {
        'Donor ID': f'D{i+1}',
        'Age': random.randint(18, 60),
        'Gender': random.choice(genders),
        'Blood Group': random.choices(blood_groups, blood_weights)[0],
        'Donated Organ': random.choices(organs, organ_weights)[0],
        'Health Status': random.choices(health_status, health_weights)[0],
        'Location': random.choices(locations, location_weights)[0],
        'HLA Typing': random.choice(hla_types)
    }
    donors.append(donor)

# Find matches and non-matches
matches = []
num_negative_samples = 1000  # Number of ineligible matches to generate
for patient in patients:
    for donor in donors:
        # Check if the pair is eligible
        is_eligible = (
            donor['Donated Organ'] == patient['Required Organ'] and
            donor['HLA Typing'] == patient['HLA Typing'] and
            patient['Blood Group'] in blood_compatibility[donor['Blood Group']] and
            donor['Age'] <= patient['Age'] + 10
        )
        if is_eligible:
            matches.append({
                'Patient ID': patient['Patient ID'],
                'Donor ID': donor['Donor ID'],
                'Organ': patient['Required Organ'],
                'Blood Group Match': f"{donor['Blood Group']} to {patient['Blood Group']}",
                'HLA Match': patient['HLA Typing'],
                'Location': patient['Location'],
                'Transplantation Eligibility': 'Yes'
            })

# Generate negative (ineligible) samples
random.shuffle(patients)
random.shuffle(donors)
for i in range(min(num_negative_samples, len(patients))):
    patient = patients[i]
    donor = donors[i]
    # Only add if the pair is not eligible
    if not (
        donor['Donated Organ'] == patient['Required Organ'] and
        donor['HLA Typing'] == patient['HLA Typing'] and
        patient['Blood Group'] in blood_compatibility[donor['Blood Group']] and
        donor['Age'] <= patient['Age'] + 10
    ):
        matches.append({
            'Patient ID': patient['Patient ID'],
            'Donor ID': donor['Donor ID'],
            'Organ': patient['Required Organ'],
            'Blood Group Match': f"{donor['Blood Group']} to {patient['Blood Group']}",
            'HLA Match': patient['HLA Typing'],
            'Location': patient['Location'],
            'Transplantation Eligibility': 'No'
        })

# Save to CSV in Google Drive
import os
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

with open(f'{output_folder}patients.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=patients[0].keys())
    writer.writeheader()
    writer.writerows(patients)

with open(f'{output_folder}donors.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=donors[0].keys())
    writer.writeheader()
    writer.writerows(donors)

with open(f'{output_folder}matches.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Patient ID', 'Donor ID', 'Organ', 'Blood Group Match', 'HLA Match', 'Location', 'Transplantation Eligibility'])
    writer.writeheader()
    writer.writerows(matches)

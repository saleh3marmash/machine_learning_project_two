import csv
from Patient import *
# Specify the path to your CSV file
csv_file_path = './Data.csv'
ages = []
genders = []
bmis = []
regions = []
num_children = []
insurance_charges = []
smoker_status = []
# Open the CSV file
patients_list = []
with open(csv_file_path, 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Create a Patient object and append it to the list
        patient = Patient(
            age=row[0],
            gender=row[1],
            bmi=row[2],
            region=row[3],
            num_children=row[4],
            insurance_charges=row[5],
            smoker=row[6]
        )
        patients_list.append(patient)

import pandas as pd

df = pd.read_csv('Soil_Moisture_Data_Filled.csv')

column_name = 'SM10'

missing_values = df[column_name].isnull().any()
missing_indices = df[df.isnull().any(axis=1)].index



if missing_values:
    print(f"The column '{column_name}' contains missing values.")
else:
    print(f"The column '{column_name}' does not contain missing values.")

print("Indices of missing values:")
print(missing_indices)
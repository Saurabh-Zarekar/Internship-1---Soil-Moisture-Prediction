import pandas as pd

file_path = 'Pune_GLDAS_data.csv'
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month

means = data.groupby(['Month', 'Day']).mean(numeric_only=True).reset_index()

data = pd.merge(data, means, on=['Month', 'Day'], suffixes=('', '_mean'))

columns_to_deseasonalize = ['SM10', 'SM30', 'Sm60', 'SM100', 'ST10', 'St30', 'St60', 'St100', 'Rain', 'Air_Temperature']

for column in columns_to_deseasonalize:
    data[f'{column}_deseasonalized'] = data[column] - data[f'{column}_mean']

deseasonalized_data = data[['Date'] + [f'{column}_deseasonalized' for column in columns_to_deseasonalize]]

output_file_path = 'C:/Desktop/IITM/Pune_GLDAS_deseasonalized_data.csv'
deseasonalized_data.to_csv(output_file_path, index=False)

print("Deseasonalized data has been saved to:", output_file_path)

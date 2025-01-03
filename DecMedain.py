import pandas as pd

data = pd.read_csv('Soil_Moisture_Data.csv')

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

data['Year'] = data['Date'].dt.year

dec_31_data = data[(data['Date'].dt.month == 12) & (data['Date'].dt.day == 31)]

median = dec_31_data.groupby('Year')['SM10'].median()

print(median)

overall_median_soil_moisture = median.median()

print("Overall median soil moisture for December 31st across all years:", overall_median_soil_moisture)

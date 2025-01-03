import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Load data
data = pd.read_csv('soil_moisture_data.csv')

# Split data into features and target variables
X = data[['Temperature', 'Surface_Soil_Moisture']]
y_surface = data['Surface_Soil_Moisture']
y_10cm = data['Soil_Moisture_10cm']
y_30cm = data['Soil_Moisture_30cm']
y_60cm = data['Soil_Moisture_60cm']
y_1m = data['Soil_Moisture_1m']

# Split data into training and testing sets
X_train, X_test, y_surface_train, y_surface_test = train_test_split(X, y_surface, test_size=0.2, random_state=42)

# Train models
model_surface = RandomForestRegressor()
model_10cm = RandomForestRegressor()
model_30cm = RandomForestRegressor()
model_60cm = RandomForestRegressor()
model_1m = RandomForestRegressor()

model_surface.fit(X_train, y_surface_train)
model_10cm.fit(X_train, y_10cm)
model_30cm.fit(X_train, y_30cm)
model_60cm.fit(X_train, y_60cm)
model_1m.fit(X_train, y_1m)

# Evaluate models
y_surface_pred = model_surface.predict(X_test)
print("Surface Soil Moisture MAE:", mean_absolute_error(y_surface_test, y_surface_pred))

# Make predictions for depths
temperature = 25  # Example temperature
surface_soil_moisture = 0.4  # Example surface soil moisture

moisture_10cm_pred = model_10cm.predict([[temperature, surface_soil_moisture]])
moisture_30cm_pred = model_30cm.predict([[temperature, surface_soil_moisture]])
moisture_60cm_pred = model_60cm.predict([[temperature, surface_soil_moisture]])
moisture_1m_pred = model_1m.predict([[temperature, surface_soil_moisture]])

print("Predicted Soil Moisture at 10cm depth:", moisture_10cm_pred)
print("Predicted Soil Moisture at 30cm depth:", moisture_30cm_pred)
print("Predicted Soil Moisture at 60cm depth:", moisture_60cm_pred)
print("Predicted Soil Moisture at 1m depth:", moisture_1m_pred)

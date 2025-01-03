import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class SoilMoisturePredictor:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.model_surface = LinearRegression()
        self.model_10cm = LinearRegression()
        self.model_20cm = LinearRegression()
        self.train_models()

    def train_models(self):
        X = self.data[['Temperature', 'Surface_Soil_Moisture']]
        y_surface = self.data['Surface_Soil_Moisture']
        y_10cm = self.data['Soil_Moisture_10cm']
        y_20cm = self.data['Soil_Moisture_20cm']

        self.model_surface.fit(X, y_surface)
        self.model_10cm.fit(X, y_10cm)
        self.model_20cm.fit(X, y_20cm)

    def evaluate_model(self, test_size=0.2, random_state=42):
        X = self.data[['Temperature', 'Surface_Soil_Moisture']]
        y_surface = self.data['Surface_Soil_Moisture']
        X_train, X_test, y_surface_train, y_surface_test = train_test_split(X, y_surface, test_size=test_size, random_state=random_state)
        y_surface_pred = self.model_surface.predict(X_test)
        return mean_absolute_error(y_surface_test, y_surface_pred)

    def predict_soil_moisture_10cm(self, surface_soil_moisture, soil_moisture_10cm, temperature):
        X = [[temperature, surface_soil_moisture, soil_moisture_10cm]]
        return self.model_10cm.predict(X)

    def predict_soil_moisture_20cm(self, surface_soil_moisture, soil_moisture_10cm, temperature):
        X = [[temperature, surface_soil_moisture, soil_moisture_10cm]]
        return self.model_20cm.predict(X)

# Instantiate the class
predictor = SoilMoisturePredictor('soil_moisture_data.csv')

# Evaluate the model
print("Surface Soil Moisture MAE:", predictor.evaluate_model())

# Make predictions
temperature = 25  # Example temperature
surface_soil_moisture = 0.4  # Example surface soil moisture
soil_moisture_10cm = 0.35  # Example soil moisture at 10 cm depth

moisture_10cm_pred = predictor.predict_soil_moisture_10cm(surface_soil_moisture, soil_moisture_10cm, temperature)
moisture_20cm_pred = predictor.predict_soil_moisture_20cm(surface_soil_moisture, soil_moisture_10cm, temperature)

print("Predicted Soil Moisture at 10cm depth:", moisture_10cm_pred)
print("Predicted Soil Moisture at 20cm depth:", moisture_20cm_pred)

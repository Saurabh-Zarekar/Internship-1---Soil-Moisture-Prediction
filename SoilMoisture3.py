import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class SoilMoisturePredictor:

    def __init__(self, FileName):

        self.data = pd.read_csv(FileName)
        self.model_surface = LinearRegression()
        self.model_10cm = LinearRegression()
        self.model_30cm = LinearRegression()
        self.model_60cm = LinearRegression()
        self.model_1m = LinearRegression()
        self.train_models()

    def train_models(self):

        #Independent variables
        X_surface = self.data[['Temperature']]
        X_10cm = self.data[['Temperature', 'Surface_Soil_Moisture']]
        X_30cm = self.data[['Temperature', 'Soil_Moisture_10cm']]
        X_60cm = self.data[['Temperature', 'Soil_Moisture_30cm']]
        X_1m = self.data[['Temperature', 'Soil_Moisture_60cm']]
        
        #Dependent Variables
        y_surface = self.data['Surface_Soil_Moisture']
        y_10cm = self.data['Soil_Moisture_10cm']
        y_30cm = self.data['Soil_Moisture_30cm']
        y_60cm = self.data['Soil_Moisture_60cm']
        y_1m = self.data['Soil_Moisture_1m']

        #Traning
        self.model_surface.fit(X_surface, y_surface)
        self.model_10cm.fit(X_10cm, y_10cm)
        self.model_30cm.fit(X_30cm, y_30cm)
        self.model_60cm.fit(X_60cm, y_60cm)
        self.model_1m.fit(X_1m, y_1m)

    def evaluate_model(self, test_size=0.2, random_state=42):

        X = self.data[['Temperature', 'Surface_Soil_Moisture']]
        y_surface = self.data['Surface_Soil_Moisture']
        X_train, X_test, y_surface_train, y_surface_test = train_test_split(X, y_surface, test_size=test_size, random_state=random_state)
        y_surface_pred = self.model_surface.predict(X_test[['Temperature']])
        return mean_absolute_error(y_surface_test, y_surface_pred)

    def predict_soil_moisture_surface(self, temperature):

        X = [[temperature]]
        return self.model_surface.predict(X)

    def predict_soil_moisture_10cm(self, temperature, surface_soil_moisture):

        X = [[temperature, surface_soil_moisture]]
        return self.model_10cm.predict(X)

    def predict_soil_moisture_30cm(self, temperature, surface_soil_moisture):

        X = [[temperature, surface_soil_moisture]]
        return self.model_30cm.predict(X)

    def predict_soil_moisture_60cm(self, temperature, surface_soil_moisture, soil_moisture_10cm):

        X = [[temperature, surface_soil_moisture, soil_moisture_10cm]]
        return self.model_60cm.predict(X)

    def predict_soil_moisture_1m(self, temperature, surface_soil_moisture, soil_moisture_10cm):

        X = [[temperature, surface_soil_moisture, soil_moisture_10cm]]
        return self.model_1m.predict(X)

def main():

    # Instantiate the class
    predictor = SoilMoisturePredictor('soil_moisture_data.csv')

    # Evaluate the model
    print("Surface Soil Moisture MAE:", predictor.evaluate_model())

    # Make predictions
    temperature = 25  # Example temperature
    surface_soil_moisture = 0.4  # Example surface soil moisture
    soil_moisture_10cm = 0.35  # Example soil moisture at 10 cm depth

    moisture_surface_pred = predictor.predict_soil_moisture_surface(temperature)
    moisture_10cm_pred = predictor.predict_soil_moisture_10cm(temperature, surface_soil_moisture)
    moisture_30cm_pred = predictor.predict_soil_moisture_30cm(temperature, surface_soil_moisture)
    moisture_60cm_pred = predictor.predict_soil_moisture_60cm(temperature, surface_soil_moisture, soil_moisture_10cm)
    moisture_1m_pred = predictor.predict_soil_moisture_1m(temperature, surface_soil_moisture, soil_moisture_10cm)

    print("Predicted Surface Soil Moisture:", moisture_surface_pred)
    print("Predicted Soil Moisture at 10cm depth:", moisture_10cm_pred)
    print("Predicted Soil Moisture at 30cm depth:", moisture_30cm_pred)
    print("Predicted Soil Moisture at 60cm depth:", moisture_60cm_pred)
    print("Predicted Soil Moisture at 1m depth:", moisture_1m_pred)

if __name__ == "__main__":
    main()

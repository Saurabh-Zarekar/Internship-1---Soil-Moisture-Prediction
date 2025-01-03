import pandas as pd
import matplotlib.pyplot as plt
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

        # Independent variables
        X_surface = self.data[['Temperature']]
        X_10cm = self.data[['Surface_Soil_Moisture']]
        X_30cm = self.data[['Soil_Moisture_10cm']]
        X_60cm = self.data[['Soil_Moisture_30cm']]
        X_1m = self.data[['Soil_Moisture_60cm']]

        # Dependent Variables
        y_surface = self.data['Surface_Soil_Moisture']
        y_10cm = self.data['Soil_Moisture_10cm']
        y_30cm = self.data['Soil_Moisture_30cm']
        y_60cm = self.data['Soil_Moisture_60cm']
        y_1m = self.data['Soil_Moisture_1m']

        # Training
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

    def predict_soil_moisture_10cm(self, surface_soil_moisture):

        X = [[surface_soil_moisture]]
        return self.model_10cm.predict(X)

    def predict_soil_moisture_30cm(self, soil_moisture_10cm):

        X = [[soil_moisture_10cm]]
        return self.model_30cm.predict(X)

    def predict_soil_moisture_60cm(self, soil_moisture_30cm):

        X = [[soil_moisture_30cm]]
        return self.model_60cm.predict(X)

    def predict_soil_moisture_1m(self, soil_moisture_60cm):

        X = [[soil_moisture_60cm]]
        return self.model_1m.predict(X)

    def plot_graph(self, x_values, y_values, depth):
        plt.plot(x_values, y_values, label=f'{depth} Depth')
        plt.xlabel('Temperature (Celsius)')
        plt.ylabel('Soil Moisture')
        plt.title(f'Soil Moisture Prediction at {depth} Depth')
        plt.legend()
        plt.show()

def validate_temperature_input():
    while True:
        try:
            temperature = float(input("Please enter the temperature (in Celsius) at which you want to predict soil moisture: "))
            if temperature < -273.15:
                print("Temperature cannot be below absolute zero. Please enter a valid temperature.")
            else:
                return temperature
        except ValueError:
            print("Invalid input. Please enter a valid temperature.")

def main():

    predictor = SoilMoisturePredictor('soil_moisture_data.csv')

    # Evaluate the model
    print("Surface Soil Moisture MAE:", predictor.evaluate_model())

    print("---------------WELCOME--------------\n")

    temperature = validate_temperature_input()

    moisture_surface_pred = predictor.predict_soil_moisture_surface(temperature)
    moisture_10cm_pred = predictor.predict_soil_moisture_10cm(moisture_surface_pred)
    moisture_30cm_pred = predictor.predict_soil_moisture_30cm(moisture_10cm_pred)
    moisture_60cm_pred = predictor.predict_soil_moisture_60cm(moisture_30cm_pred)
    moisture_1m_pred = predictor.predict_soil_moisture_1m(moisture_60cm_pred)

    depths = ['Surface', '10cm', '30cm', '60cm', '1m']
    predictions = [moisture_surface_pred, moisture_10cm_pred, moisture_30cm_pred, moisture_60cm_pred, moisture_1m_pred]

    for depth, prediction in zip(depths, predictions):
        predictor.plot_graph(temperature, prediction, depth)

if __name__ == "__main__":
    main()

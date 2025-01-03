import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class SoilMoisturePredictor:

    def __init__(self, FileName):

        self.data = pd.read_csv(FileName)
        self.model = self.build_model()
        self.train_models()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=[1]),  # Input layer
            tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
            tf.keras.layers.Dense(1)  # Output layer
        ])

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        return model

    def train_models(self):
        # Independent variable
        X = self.data[['Temperature']]
        # Dependent Variable
        y = self.data['Surface_Soil_Moisture']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, epochs=100, verbose=0)  # Train the model

    def evaluate_model(self):
        X_test = self.data[['Temperature']]
        y_test = self.data['Surface_Soil_Moisture']
        predictions = self.model.predict(X_test).flatten()
        return mean_absolute_error(y_test, predictions)

    def predict_soil_moisture_surface(self, temperature):
    # Reshape the temperature value to match the model input shape
        temperature = np.array([[temperature]])  # Convert to a 2D array
        return self.model_surface.predict(temperature)[0]  # Predict and return the result


def main():
    predictor = SoilMoisturePredictor('Soil_Data.csv')

    # Evaluate the model
    print("Surface Soil Moisture MAE:", predictor.evaluate_model())

    print("---------------WELCOME--------------\n")
    print("--------------This application is used to predict soil moisture-------------")

    temperatures = np.arange(-10, 41, 5)  # Store temperatures for plotting
    moisture_surface_pred_list = []

    for temperature in temperatures:  # Loop over a range of temperatures
        moisture_surface_pred = predictor.predict_soil_moisture_surface(temperature)
        moisture_surface_pred_list.append(moisture_surface_pred)

    print("Temperature:", temperatures)
    print("Predicted Surface Soil Moisture:", moisture_surface_pred_list)

    plt.plot(temperatures, moisture_surface_pred_list, 'ro')  # Plot predicted soil moisture
    plt.xlabel('Temperature (Celsius)')
    plt.ylabel('Soil Moisture')
    plt.title('Soil Moisture Prediction at Surface Depth')
    plt.show()

if __name__ == "__main__":
    main()

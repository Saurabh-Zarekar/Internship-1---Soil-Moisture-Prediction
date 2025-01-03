import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class SoilMoisturePredictor:

    def __init__(self, FileName):

        self.data = pd.read_csv(FileName)
        self.model_surface = LinearRegression()
        self.model_30cm = LinearRegression()
        self.model_60cm = LinearRegression()
        self.model_1m = LinearRegression()
        self.train_models()

    def train_models(self):

        # Independent variables
        X_surface = self.data[['Temperature']]
        X_30cm = self.data[['Surface_Soil_Moisture']]
        X_60cm = self.data[['Soil_Moisture_30cm']]
        X_1m = self.data[['Soil_Moisture_60cm']]

        # Dependent Variables
        y_surface = self.data['Surface_Soil_Moisture']
        y_30cm = self.data['Soil_Moisture_30cm']
        y_60cm = self.data['Soil_Moisture_60cm']
        y_1m = self.data['Soil_Moisture_1m']

        # Training
        self.model_surface.fit(X_surface, y_surface)
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
        X = pd.DataFrame({'Temperature': [temperature]})
        return self.model_surface.predict(X)

    def predict_soil_moisture_30cm(self, moisture_surface):
        X = pd.DataFrame({'Surface_Soil_Moisture': [moisture_surface]})
        return self.model_30cm.predict(X)

    def predict_soil_moisture_60cm(self, moisture_30cm):
        X = pd.DataFrame({'Soil_Moisture_30cm': [moisture_30cm]})
        return self.model_60cm.predict(X)

    def predict_soil_moisture_1m(self, moisture_60cm):
        X = pd.DataFrame({'Soil_Moisture_60cm': [moisture_60cm]})
        return self.model_1m.predict(X)


def plot_graph(x_values, y_values, depth, regression_line_input, regression_line_predicted):
    plt.plot(x_values, regression_line_input, 'b-', label=f'Regression Line (Input Soil Moisture)')  # Plot regression line for input soil moisture in blue
    plt.plot(x_values, y_values, 'ro', label=f'Predicted {depth} Moisture')  # Plot predicted soil moisture in red
    plt.plot(x_values, regression_line_predicted, 'r-', label=f'Regression Line (Predicted Soil Moisture)')  # Plot regression line for predicted soil moisture in red
    plt.xlabel('Temperature (Celsius)')
    plt.ylabel('Soil Moisture')
    plt.title(f'Soil Moisture Prediction at {depth} Depth')
    plt.legend()
    plt.show()


def main():
    predictor = SoilMoisturePredictor('Soil_Data.csv')

    # Evaluate the model
    print("Surface Soil Moisture MAE:", predictor.evaluate_model())

    print("---------------WELCOME--------------\n")
    print("--------------This application is used to predict soil moisture-------------")

    temperatures = []  # Store temperatures for plotting
    moisture_surface_pred_list = []
    moisture_30cm_pred_list = []
    moisture_60cm_pred_list = []
    moisture_1m_pred_list = []

    for temperature in range(-10, 41, 5):  # Loop over a range of temperatures
        temperatures.append(temperature)

        moisture_surface_pred = predictor.predict_soil_moisture_surface(temperature)
        moisture_30cm_pred = predictor.predict_soil_moisture_30cm(moisture_surface_pred)
        moisture_60cm_pred = predictor.predict_soil_moisture_60cm(moisture_30cm_pred)
        moisture_1m_pred = predictor.predict_soil_moisture_1m(moisture_60cm_pred)

        moisture_surface_pred_list.append(moisture_surface_pred[0])
        moisture_30cm_pred_list.append(moisture_30cm_pred[0])
        moisture_60cm_pred_list.append(moisture_60cm_pred[0])
        moisture_1m_pred_list.append(moisture_1m_pred[0])

    print("Temperature:", temperatures)
    print("Predicted Surface Soil Moisture:", moisture_surface_pred_list)
    print("Predicted Soil Moisture at 30cm depth:", moisture_30cm_pred_list)
    print("Predicted Soil Moisture at 60cm depth:", moisture_60cm_pred_list)
    print("Predicted Soil Moisture at 1m depth:", moisture_1m_pred_list)

    depths = ['Surface', '30cm', '60cm', '1m']
    predictions = [moisture_surface_pred_list, moisture_30cm_pred_list, moisture_60cm_pred_list, moisture_1m_pred_list]

    for depth, prediction in zip(depths, predictions):
        # Fit a regression line using numpy polyfit for both input and predicted soil moisture
        regression_line_input = np.poly1d(np.polyfit(temperatures, predictor.data[f'{depth}_Soil_Moisture'], 1))
        regression_line_predicted = np.poly1d(np.polyfit(temperatures, prediction, 1))
        plot_graph(temperatures, prediction, depth, regression_line_input(temperatures), regression_line_predicted(temperatures))


if __name__ == "__main__":
    main()

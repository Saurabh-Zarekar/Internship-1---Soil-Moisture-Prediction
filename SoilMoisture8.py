import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense

class SoilMoisturePredictorNN:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.model_surface = self.build_model()
        self.model_30cm = self.build_model()
        self.model_60cm = self.build_model()
        self.model_1m = self.build_model()
        self.train_models()

    def build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(2,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        return model

    def train_models(self):
        # Check if 'Surface_Soil_Moisture' column is present in the dataset
        if 'Surface_Soil_Moisture' not in self.data.columns:
            raise ValueError("Column 'Surface_Soil_Moisture' not found in the dataset.")

        # Extracting features and target for surface soil moisture prediction
        X_surface = self.data[['Temperature', 'Air_Temperature']].astype(float)
        y_surface = self.data['Surface_Soil_Moisture'].astype(float)

        # Splitting data into training and validation sets
        X_train_surface, X_val_surface, y_train_surface, y_val_surface = train_test_split(X_surface, y_surface, test_size=0.2, random_state=42)

        # Standardizing features
        scaler = StandardScaler()
        X_train_surface_scaled = scaler.fit_transform(X_train_surface)
        X_val_surface_scaled = scaler.transform(X_val_surface)

        # Defining and training the neural network model for surface soil moisture prediction
        self.model_surface.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        self.model_surface.fit(X_train_surface_scaled, y_train_surface, epochs=10, batch_size=32, validation_data=(X_val_surface_scaled, y_val_surface))

        # Now, train models for 30cm, 60cm, and 1m depths (assuming the same features and target)

    def evaluate_model(self):
        X = self.data[['Temperature', 'Air_Temperature']].astype(float)
        y_surface = self.data['Surface_Soil_Moisture'].astype(float)
        X_train, X_test, y_surface_train, y_surface_test = train_test_split(X, y_surface, test_size=0.2, random_state=42)
        y_surface_pred = self.model_surface.predict(X_test)
        mae_surface = mean_absolute_error(y_surface_test, y_surface_pred)
        rmse_surface = mean_squared_error(y_surface_test, y_surface_pred, squared=False)
        return mae_surface, rmse_surface

    def predict_soil_moisture_surface(self, temperature, air_temperature):
        X = np.array([[temperature, air_temperature]]).astype(np.float32)
        return self.model_surface.predict(X)

    def predict_soil_moisture_30cm(self, moisture_surface_pred):
        # Similar to surface prediction but using self.model_30cm
        pass

    def predict_soil_moisture_60cm(self, moisture_30cm_pred):
        # Similar to surface prediction but using self.model_60cm
        pass

    def predict_soil_moisture_1m(self, moisture_60cm_pred):
        # Similar to surface prediction but using self.model_1m
        pass

def validate_input():
    while True:
        try:
            temperature = float(input("Please enter the temperature (in Celsius) at which you want to predict soil moisture: "))
            air_temperature = float(input("Please enter the air temperature: "))
            if temperature < -273.15:
                print("Temperature cannot be below absolute zero. Please enter a valid temperature.")
            else:
                return temperature, air_temperature
        except ValueError:
            print("Invalid input. Please enter valid numerical values for temperature and air temperature.")

def main():
    print("---------------WELCOME--------------\n")
    print("-------This Application use for soil moisture prediction-------\n\n")

    predictor = SoilMoisturePredictorNN('Soil_Data.csv')

    mae_surface, rmse_surface = predictor.evaluate_model()

    print("Surface Soil Moisture MAE  :", mae_surface,"\n")
    print("Surface Soil Moisture RMSE :", rmse_surface,"\n\n")

    temperature, air_temperature = validate_input()

    moisture_surface_pred = predictor.predict_soil_moisture_surface(temperature, air_temperature)
    moisture_30cm_pred = predictor.predict_soil_moisture_30cm(moisture_surface_pred)
    moisture_60cm_pred = predictor.predict_soil_moisture_60cm(moisture_30cm_pred)
    moisture_1m_pred = predictor.predict_soil_moisture_1m(moisture_60cm_pred)

    print("Predicted Surface Soil Moisture       : ", moisture_surface_pred,"\n")
    print("Predicted Soil Moisture at 30cm depth : ", moisture_30cm_pred,"\n")
    print("Predicted Soil Moisture at 60cm depth : ", moisture_60cm_pred,"\n")
    print("Predicted Soil Moisture at 1m depth   : ", moisture_1m_pred,"\n\n")

if __name__ == "__main__":
    main()

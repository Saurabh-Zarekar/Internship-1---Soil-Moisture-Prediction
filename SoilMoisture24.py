import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data2.csv')

    X = data[['Rain', 'Air_Temperature', 'St10','Rainref']]  
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)  
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)

    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP))
    print("R-squared (R2): ", r2_score(Y_test, SMP))


def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

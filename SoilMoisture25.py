from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data2.csv')

    X = data[['Rain', 'Air_Temperature', 'St10']]  
    Y_SM = data['SM10']

    poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
    X_poly = poly.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y_SM, test_size=0.2, random_state=42)

    model = LinearRegression() 
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, Y_pred))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, Y_pred))
    print("R-squared (R2): ", r2_score(Y_test, Y_pred))

    # Print the coefficients of the polynomial regression equation
    print("Polynomial Regression Equation:")
    print("Intercept:", model.intercept_)
    print("Coefficients:")
    for i, coef in enumerate(model.coef_):
        print(f"X{i}: {coef}")

if __name__ == "__main__":
    SoilMoisture()

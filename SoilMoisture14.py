import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data2.csv')

    X = data[['Rain', 'ref', 'St10']]  
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)

    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP))
    print("R-squared (R2): ", r2_score(Y_test, SMP))
    

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()
#98
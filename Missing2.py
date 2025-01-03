import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def fill_missing(data):
    
    data['Air_Temperature'] = data['Air_Temperature'].replace(' ....', np.nan)


    missing_data = data[data['Air_Temperature'].isnull()]
    complete_data = data.dropna()

    
    independent = ['ST10']
    dependent = 'Air_Temperature'

    
    X_train = complete_data[independent]
    y_train = complete_data[dependent]

    
    model = LinearRegression()
    model.fit(X_train, y_train)

    X_missing = missing_data[independent]
    predicted_values = model.predict(X_missing)

    
    data.loc[data['Air_Temperature'].isnull(), 'Air_Temperature'] = predicted_values

    y_true = complete_data[dependent]  # Actual values from complete data
    y_pred = model.predict(X_train)    # Predicted values from the model

    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    return data

def ReadCSV():
    
    data = pd.read_csv('Pune_GLDAS_data_Saurabh (3).csv')

    
    data = fill_missing(data)

    
    data.to_csv('Pune_GLDAS_data.csv', index=False)

    print("DataFrame saved to Soil_Moisture_Data_Filled.csv")



def main():
    ReadCSV()

if __name__=="__main__":
    main()

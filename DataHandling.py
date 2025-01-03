import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def FillData():

    data = pd.read_csv('Soil_Moisture_Data.csv')

    data['Air_Temperature'] = data['Air_Temperature'].replace(' ....',np.nan)

    missing_data = data[data['Air_Temperature'].isnull()]
    complete_data = data.dropna()

    X_Train = complete_data[['St10']]
    Y_Train = complete_data['Air_Temperature']

    model = LinearRegression()
    model.fit(X_Train, Y_Train)

    X_missing = missing_data[['St10']]
    X_predict = model.predict(X_missing)

    data.loc[data['Air_Temperature'].isnull(), 'Air_Temperature'] = X_predict

    Y_true = complete_data['Air_Temperature']
    Y_predict = model.predict(X_Train)

    mae = mean_absolute_error(Y_true,Y_predict)
    mse = mean_squared_error(Y_true,Y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_true,Y_predict)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    return data


def main():
    data = FillData()
    
    data.to_csv('Soil_Moisture_Data2.csv', index=False)
    print("DataFrame saved to Soil_Moisture_Data2.csv")

if __name__=="__main__":
    main()
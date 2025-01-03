import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_accuracy(y_true, y_pred):
    total_error = np.sum(np.abs(y_true - y_pred))
    accuracy = 1 - (total_error / np.sum(y_true))
    return accuracy * 100  # Convert to percentage


def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data.csv')

    data.dropna(subset=['SM10'], inplace=True)

    X1 = data[[ 'Rain','St10', 'Air_Temperature','SM10']]
    X2 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1']]
    X3 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1','ref2']]
    X4 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1','ref2','ref3']]
    X5 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1','ref2','ref3','ref4']]
    X6 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1','ref2','ref3','ref4','ref5']]
    X7 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1','ref2','ref3','ref5','ref6','ref4']]
    X8 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1','ref2','ref3','ref4','ref5','ref6','ref7']]
    X9 = data[[ 'Rain','St10', 'Air_Temperature','SM10','ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9']]
    X10= data[[ 'Rain','St10', 'Air_Temperature','SM10',,'ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10']]

    Y_SM1 = data['ref1']
    Y_SM2 = data['ref2']
    Y_SM3 = data['ref3']
    Y_SM4 = data['ref4']
    Y_SM5 = data['ref5']
    Y_SM6 = data['ref6']
    Y_SM7 = data['ref7']
    Y_SM8 = data['ref8']
    Y_SM9 = data['ref9']
    Y_SM10 = data['ref10']

    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y_SM1, test_size=0.2, random_state=42)
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y_SM2, test_size=0.2, random_state=42)
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y_SM3, test_size=0.2, random_state=42)
    X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X4, Y_SM4, test_size=0.2, random_state=42)
    X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X5, Y_SM5, test_size=0.2, random_state=42)
    X_train6, X_test6, Y_train6, Y_test6 = train_test_split(X6, Y_SM6, test_size=0.2, random_state=42)
    X_train7, X_test7, Y_train7, Y_test7 = train_test_split(X7, Y_SM7, test_size=0.2, random_state=42)
    X_train8, X_test8, Y_train8, Y_test8 = train_test_split(X8, Y_SM8, test_size=0.2, random_state=42)
    X_train9, X_test9, Y_train9, Y_test9 = train_test_split(X9, Y_SM9, test_size=0.2, random_state=42)
    X_train10, X_test10, Y_train10, Y_test10 = train_test_split(X10, Y_SM10, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train1, Y_train1)
    SMP = model.predict(X_test1)

    model.fit(X_train2, Y_train2)
    SMP = model.predict(X_test2)

    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP))
    print("R-squared (R2): ", r2_score(Y_test, SMP))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP))
    print("Root Mean Squared Error:", rmse)
    
    accuracy = calculate_accuracy(Y_test, SMP)
    print("Accuracy: {:.2f}%".format(accuracy))


def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

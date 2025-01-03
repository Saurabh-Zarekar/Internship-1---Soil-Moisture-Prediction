import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_accuracy(y_true, y_pred):
    total_error = np.sum(np.abs(y_true - y_pred))
    accuracy = 1 - (total_error / np.sum(y_true))
    return accuracy * 100  # Convert to percentage


def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data_Filled.csv')

    data.dropna(subset=['SM10'], inplace=True)

    X = data[['Rain', 'St10', 'Air_Temperature','ref1']]
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)

    print("Predicting Same Day Using Previous Day SM referance : ")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP))
    print("R-squared (R2): ", r2_score(Y_test, SMP))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP)
    print("Accuracy: {:.2f}%".format(accuracy))
    
    data['SMP'] = model.predict(X)

    print("----------------------------------------------------------------------")

    X2 = data[['Rain', 'St10', 'Air_Temperature', 'SMP']]
    Y_SM2 = data['next1']

    X_train, X_test, Y_train, Y_test = train_test_split(X2, Y_SM2, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP2 = model.predict(X_test)

    print("Predicting 1st next Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP2))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP2))
    print("R-squared (R2): ", r2_score(Y_test, SMP2))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP2))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP2)
    print("Accuracy: {:.2f}%".format(accuracy))
    
    data['SMP2'] = pd.Series(model.predict(X_test))

    print("----------------------------------------------------------------------")

    X3 = data[['Rain', 'St10', 'Air_Temperature', 'SMP', 'SMP2']]
    Y_SM3 = data['next2']

    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y_SM3, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP3 = model.predict(X_test)

    print("Predicting 2nd Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP3))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP3))
    print("R-squared (R2): ", r2_score(Y_test, SMP3))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP3))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP3)
    print("Accuracy: {:.2f}%".format(accuracy))
    
    data['SMP3'] = pd.Series(model.predict(X_test))

    print("----------------------------------------------------------------------")

    X4 = data[['Rain', 'St10', 'Air_Temperature', 'SMP', 'SMP2', 'SMP3']]
    Y_SM4 = data['next3']

    X_train, X_test, Y_train, Y_test = train_test_split(X4, Y_SM4, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP4 = model.predict(X_test)

    print("Predicting 3rd Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP4))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP4))
    print("R-squared (R2): ", r2_score(Y_test, SMP4))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP4))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP4)
    print("Accuracy: {:.2f}%".format(accuracy))
    
    data['SMP4'] = pd.Series(model.predict(X_test))

    print("----------------------------------------------------------------------")

    X5 = data[['Rain', 'St10', 'Air_Temperature', 'SMP', 'SMP2', 'SMP3','SMP4']]
    Y_SM5 = data['next4']

    X_train, X_test, Y_train, Y_test = train_test_split(X5, Y_SM5, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP5 = model.predict(X_test)

    print("Predicting 4th Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP5))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP5))
    print("R-squared (R2): ", r2_score(Y_test, SMP5))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP5))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP5)
    print("Accuracy: {:.2f}%".format(accuracy))

    data['SMP5'] = pd.Series(model.predict(X_test))

    print("----------------------------------------------------------------------")

    X6 = data[['Rain', 'St10', 'Air_Temperature', 'SMP', 'SMP2', 'SMP3','SMP4','SMP5']]
    Y_SM6 = data['next5']

    X_train, X_test, Y_train, Y_test = train_test_split(X6, Y_SM6, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP6 = model.predict(X_test)

    print("Predicting 5th Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP6))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP6))
    print("R-squared (R2): ", r2_score(Y_test, SMP6))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP6))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP6)
    print("Accuracy: {:.2f}%".format(accuracy))

    data['SMP6'] = pd.Series(model.predict(X_test))

    print("----------------------------------------------------------------------")

    X7 = data[['Rain', 'St10', 'Air_Temperature', 'SMP', 'SMP2', 'SMP3','SMP4','SMP5','SMP6']]
    Y_SM7 = data['next6']

    X_train, X_test, Y_train, Y_test = train_test_split(X7, Y_SM7, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP7 = model.predict(X_test)

    print("Predicting 6th Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP7))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP7))
    print("R-squared (R2): ", r2_score(Y_test, SMP7))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP7))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP7)
    print("Accuracy: {:.2f}%".format(accuracy))

    data['SMP7'] = pd.Series(model.predict(X_test))

    print("----------------------------------------------------------------------")

    X8 = data[['Rain', 'St10', 'Air_Temperature', 'SMP', 'SMP2', 'SMP3','SMP4','SMP5','SMP6','SMP7']]
    Y_SM8 = data['next7']

    X_train, X_test, Y_train, Y_test = train_test_split(X8, Y_SM8, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP8 = model.predict(X_test)

    print("Predicting 7th Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP8))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP8))
    print("R-squared (R2): ", r2_score(Y_test, SMP8))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP8))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP8)
    print("Accuracy: {:.2f}%".format(accuracy))

    data['SMP8'] = pd.Series(model.predict(X_test))

    print("----------------------------------------------------------------------")

    X9 = data[['Rain', 'St10', 'Air_Temperature', 'SMP', 'SMP2', 'SMP3','SMP4','SMP5','SMP6','SMP7','SMP8']]
    Y_SM9 = data['next8']

    X_train, X_test, Y_train, Y_test = train_test_split(X9, Y_SM9, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP9 = model.predict(X_test)

    print("Predicting 8th Day:")
    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP9))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP9))
    print("R-squared (R2): ", r2_score(Y_test, SMP9))
    rmse = np.sqrt(mean_squared_error(Y_test, SMP9))
    print("Root Mean Squared Error:", rmse)
    accuracy = calculate_accuracy(Y_test, SMP9)
    print("Accuracy: {:.2f}%".format(accuracy))


def main():
    SoilMoisture()


if __name__ == "__main__":
    main()
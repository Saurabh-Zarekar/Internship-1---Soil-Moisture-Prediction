import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def calculate_accuracy(y_true, y_pred):
    total_error = np.sum(np.abs(y_true - y_pred))
    accuracy = 1 - (total_error / np.sum(y_true))
    return accuracy * 100  # Convert to percentage


def SoilMoisture():
    data = pd.read_csv('SM1_Data.csv')


    X = data[['Rain', 'St10', 'Air_Temperature','ref40']]
    Y_SM = data['SM10']

    print(X.dtypes)
    print(Y_SM.dtype)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, Y_pred))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, Y_pred))
    print("R-squared (R2): ", r2_score(Y_test, Y_pred))
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    print("Root Mean Squared Error:", rmse)

    accuracy = calculate_accuracy(Y_test, Y_pred)
    print("Accuracy: {:.2f}%".format(accuracy))

    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    features = X.columns[sorted_idx]
    plt.barh(features, feature_importance[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.show()


def main():
    SoilMoisture()


if __name__ == "__main__":
    main()

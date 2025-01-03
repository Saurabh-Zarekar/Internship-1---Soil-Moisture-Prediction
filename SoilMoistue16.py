import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data2.csv')

    X = data[['Rain', 'ref', 'St10']]  
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)  # Initialize Random Forest model
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)

    print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP))
    print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP))
    print("R-squared (R2): ", r2_score(Y_test, SMP))
    print("Mean Absolute Percentage Error (MAPE): ", mean_absolute_percentage_error(Y_test, SMP))

    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    print("\nFeature Importance:")
    print(feature_importance)

    # Plotting feature importance
    feature_importance.plot(kind='bar')
    plt.title('Feature Importance in Random Forest')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.show()

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

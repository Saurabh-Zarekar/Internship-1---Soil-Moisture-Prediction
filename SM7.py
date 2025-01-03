import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_accuracy(y_true, y_pred):
    total_error = np.sum(np.abs(y_true - y_pred))
    accuracy = 1 - (total_error / np.sum(y_true))
    return accuracy * 100  

def SoilMoisture():
    data = pd.read_csv('SM1_Data.csv')

    X = data[[ 'Rain','St10', 'Air_Temperature','ref40']]
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)

    index_range = range(len(SMP))

    print("Y_test:\n", Y_test)
    print("\nY_test index (first 50):\n", Y_test.index[:50])

    plt.plot(index_range, SMP, label='Predicted Y_test', color='black')
    plt.plot(Y_test.index[:50], Y_test[:50], label='Actual Y_test', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Soil Moisture')
    plt.title('Actual vs Predicted Soil Moisture for Y_test')
    plt.legend()
    plt.xlim(0, 50)  
    plt.show()

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()


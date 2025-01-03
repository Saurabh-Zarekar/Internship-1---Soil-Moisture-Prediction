import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def create_lag_features(data, lag):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'SM10_lag_{i}'] = lagged_data['SM10'].shift(i)
    return lagged_data.dropna()

def forecast_with_random_forest(data, lag, forecast_steps):
    lagged_data = create_lag_features(data, lag)
    
    X = lagged_data.drop(columns=['SM10'])
    y = lagged_data['SM10']
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    forecast = []
    last_window = X_train[-lag:].values.reshape(1, -1)  
    for i in range(forecast_steps):
        prediction = model.predict(last_window)[0]
        forecast.append(prediction)
        last_window = np.roll(last_window, -1)  
        last_window[0, -1] = prediction 
    
    return forecast

def main():
    data = pd.read_csv('SM1_Data.csv')
    
    lag = 12  
    forecast_steps = 48  

    forecast = forecast_with_random_forest(data, lag, forecast_steps)
    
    plt.plot(range(1, forecast_steps + 1), forecast, label='Forecasted Data', color='red')
    plt.xlabel('Time')
    plt.ylabel('Soil Moisture')
    plt.title('Forecasted Soil Moisture for Next 4 Years using Random Forest')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

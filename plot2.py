import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('SM1_Data.csv', index_col='Date', parse_dates=True)

X = data[['Air_Temperature','St10','Rain','ref40']] 
Y = data['SM10']

train_data = data.iloc[2:13150]  
test_data = data.iloc[13150:15310]  
X_train = train_data[['Air_Temperature','St10','Rain','ref40']]  
Y_train = train_data['SM10']
X_test = test_data[['Air_Temperature','St10','Rain','ref40']] 
Y_test = test_data['SM10']

model = RandomForestRegressor(max_depth=7, random_state=42)
model.fit(X_train, Y_train)

SMP = model.predict(X_test)

print("Mean Absolute Error (MAE): ", mean_absolute_error(Y_test, SMP))
print("Mean Squared Error (MSE): ", mean_squared_error(Y_test, SMP))
print("R-squared (R2): ", r2_score(Y_test, SMP))
rmse = np.sqrt(mean_squared_error(Y_test, SMP))
print("Root Mean Squared Error:", rmse)

residuals = Y_test - SMP
residuals_std = np.std(residuals)
n = len(Y_test)
std_error_prediction = residuals_std / np.sqrt(n)
critical_value = 1.96  
prediction_interval_lower = SMP - critical_value * std_error_prediction
prediction_interval_upper = SMP + critical_value * std_error_prediction

print("Prediction Interval Lower Bound:", prediction_interval_lower)
print("Prediction Interval Upper Bound:", prediction_interval_upper)

plt.plot(Y_test.index, SMP, color='red', label='Predicted SMP')
plt.xlim(0,365)
plt.fill_between(Y_test.index, prediction_interval_lower, prediction_interval_upper, color='lightgrey', label='Prediction Interval')
plt.legend()
plt.show()



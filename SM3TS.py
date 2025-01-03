import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data.csv')

    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%y')
    data['Year'] = data['Date'].dt.year
    print(data[['Date', 'Year']])

    X = data[[ 'Rain','St10', 'Air_Temperature','ref50']]
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)
    
    i=0
    j=365

    while(j < 15309):
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=X_test, x=X_test.index, y=SMP, label='Predicted Surface Soil Moisture', color='blue', alpha=0.6)
        sns.lineplot(data=data, x=data.index, y='SM10', label='Observed Soil Moisture', color='red')
        plt.xlabel('values')
        plt.xlim(i,j)
        plt.ylabel('Soil Moisture')
        plt.title('Time Series Plot of Features')
        plt.legend()
        plt.show()
        i=j
        j=j+365
    

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

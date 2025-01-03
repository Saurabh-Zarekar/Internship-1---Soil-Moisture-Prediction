import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data2.csv')

    X = data[['Rain', 'Air_Temperature', 'St10']]  
    Y_SM = data['SM10']

    SMP = []  
    depths = []
    r_squ = []  

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    for i in range(1, 20):
        model = RandomForestRegressor(max_depth=i, random_state=42) 
        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)
        SMP.append(predictions)
        depths.append(i)
        r_squ.append(r2_score(Y_test, predictions))

    print(SMP)
    print(depths)
    print(r_squ)

    plot_time_series(depths, r_squ)

def plot_time_series(depths, r_squ):
    plt.bar(depths,r_squ)
    plt.show()

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()
#2.5 >= rainyday
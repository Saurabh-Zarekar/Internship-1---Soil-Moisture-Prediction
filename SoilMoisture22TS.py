import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def SoilMoisture():

    data = pd.read_csv('Soil_Moisture_Data2.csv')

    X = data[['Rain', 'Air_Temperature', 'St10', 'Rainref']]  
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.5, random_state=42)

    model = RandomForestRegressor(max_depth=10, random_state=42) 
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)
    
    
    sns.set(style="whitegrid")  
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=X_test, x=X_test.index, y=SMP, label='Predicted Surface Soil Moisture', color='blue', alpha=0.6)
    sns.lineplot(data=data, x=data.index, y='SM10', label='Observed SOil Moisture', color='red')
    
    print(Y_test)
    #data["SM10"].plot()
    plt.xlabel('Values')
    plt.xlim(0,2000)
    plt.ylabel('Soil Moisture')
    plt.title('Time Series Plot of Features')
    plt.legend()
    plt.show()


def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

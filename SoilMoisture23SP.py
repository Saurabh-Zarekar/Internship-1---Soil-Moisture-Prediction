import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def SoilMoisture():
    df = pd.read_csv('data.csv')
    data = pd.read_csv('Soil_Moisture_Data2.csv')

    X = data[['Rain', 'Air_Temperature', 'St10', 'Rainref']]  
    Y_SM = data['SM10']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.5, random_state=42)

    model = RandomForestRegressor(max_depth=10, random_state=42) 
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)
    
    sns.set(style="whitegrid")  
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y_test, y=SMP, color='blue', label='Predicted')
    sns.scatterplot(x=Y_test, y=Y_test, color='red', label='Actual')

    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.title('Scatter Plot of Soil Moisture')
    plt.legend()
    plt.show()

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

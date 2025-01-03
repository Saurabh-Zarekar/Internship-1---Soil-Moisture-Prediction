import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

def SoilMoisture():
    df = pd.read_csv('data.csv')
    
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    
    df.set_index('Date', inplace=True)
    
    sns.set(style="whitegrid") 

   
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='SM10', label='Surface Soil Moisture', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Soil Moisture')
    plt.title('Surface Soil Moisture over Time')
    plt.show()

    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index, y='Soil_Temperature', label='Soil Temperature', color='red')
    sns.lineplot(data=df, x=df.index, y='Air_Temperature', label='Air Temperature', color='green')
    sns.lineplot(data=df, x=df.index, y='Rainfall', label='Rainfall', color='purple')
   
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Plot of Features')
    plt.show()

    
    plot_acf(df['SM10'])
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Plot of Surface Soil Moisture')
    plt.show()

   
    result = adfuller(df['SM10'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

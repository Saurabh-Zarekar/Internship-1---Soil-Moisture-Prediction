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
    #sns.lineplot(data=df, x=df.index, y='ref', label='Privious day', color='yellow')
    #sns.lineplot(data=df, x=df.index, y='St10', label='Soil Temperature', color='red')
    #sns.lineplot(data=df, x=df.index, y='Air_Temperature', label='Air Temperature', color='green')
    #sns.lineplot(data=df, x=df.index, y='Rain', label='Rainfall', color='purple')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Plot of Features')
    plt.legend()
    plt.show()


def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

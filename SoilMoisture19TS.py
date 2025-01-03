import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller



def SoilMoisture():
    df = pd.read_csv('data.csv')
    print(df.head())

    
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6)) 
    sns.lineplot(data=df, x='Date', y='SM10', label='SM10', color='blue')

   
    plt.xlabel('Date')
    plt.ylabel('SM')
    plt.title('Surface soil moisture')

    plt.show()


def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

import pandas as pd


def ReadCSV():
    data=pd.read_csv('Soil_Moisture_Data.csv')
    print(data.info()) #info about dataframes
    print("-----------------------------\n")
    print(data.isnull().sum())
    print("-----------------------")
    print((data['Air_Temperature'] == ' ....').sum())

    sumo = 0

    for index, i in data.iterrows():
        Air_Temperature = i['Air_Temperature']
        if Air_Temperature == ' ....':
            sumo+=1

    print("Total values missing are : ",sumo)


def main():
    ReadCSV()

if __name__=="__main__":
    main()
import pandas as pd


def ReadCSV():
    data=pd.read_csv('Soil_Moisture_Data_Filled.csv')
    print(data.info()) #info about dataframes
    print("-----------------------------\n")
    print(data.isnull().sum())
    print("-----------------------")
    print((data['SM10'] == ' ....').sum())

    sumo = 0

    for index, i in data.iterrows():
        SM10 = i['SM10']
        if SM10 == ' ....':
            sumo+=1

    print("Total values missing are : ",sumo)


def main():
    ReadCSV()

if __name__=="__main__":
    main()
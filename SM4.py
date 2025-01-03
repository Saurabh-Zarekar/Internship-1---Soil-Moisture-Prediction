import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def FillData():
    data = pd.read_csv('SM1_Data.csv')

    missing_data = data[data['ref60'].isnull()]
    complete_data = data.dropna(subset=['ref60'])

    X_train = complete_data[['SM10']]
    y_train = complete_data['ref60']

    model = RandomForestRegressor(max_depth=7,random_state=42)
    model.fit(X_train, y_train)

    X_missing = missing_data[['SM10']]
    y_predict = model.predict(X_missing)

    data.loc[data['ref60'].isnull(), 'ref60'] = y_predict

    mae = mean_absolute_error(y_train, model.predict(X_train))
    mse = mean_squared_error(y_train, model.predict(X_train))
    rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)
    r2 = r2_score(y_train, model.predict(X_train))

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    return data

def main():
    data = FillData()

    data.to_csv('Data2.csv', index=False)
    print("DataFrame saved to Data3.csv")

if __name__ == "__main__":
    main()

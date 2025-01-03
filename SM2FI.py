import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def SoilMoisture():
    data = pd.read_csv('Soil_Moisture_Data.csv')

    data.dropna(subset=['SM10'], inplace=True)

    X = data[['Rain','Soil Temperature', 'Air_Temperature','Ref40']]
    Y_SM = data['SM10']

    print(X.dtypes)
    print(Y_SM.dtype)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_SM, test_size=0.2, random_state=42)

    model = RandomForestRegressor(max_depth=7, random_state=42)
    model.fit(X_train, Y_train)

    SMP = model.predict(X_test)

    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    print("\nFeature Importance:")
    print(feature_importance)

    colors = sns.color_palette('deep')[0:len(feature_importance)]
    feature_importance.plot(kind='pie', colors=colors, autopct='%1.1f%%')
    plt.title('Feature Importance in Random Forest Regression')
    plt.show()

def main():
    SoilMoisture()

if __name__ == "__main__":
    main()

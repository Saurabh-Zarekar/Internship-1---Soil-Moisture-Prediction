import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('Pune_GLDAS_deseasonalized_data.csv')

corr = data[['SM10', 'ST10','Rain','Air_Temperature']].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='magma', fmt=".2f")
plt.title('Correlation Heatmap between Variables')
plt.show()

import pandas as pd

data = pd.read_csv('Pune_GLDAS_data.csv')

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month

mean_data_SM = {}
mean_data_St = {}
mean_data_At = {}
mean_data_R = {}
mean_data_ref = {}

Year = 1980
while Year != 2021:
    for i in range(1, 13):
        for j in range(1, 32):  
            filtered_data = data[(data['Month'] == i) & (data['Day'] == j) & (data['Date'].dt.year == Year)]
            
            if not filtered_data.empty:
                mean_data_SM[(j, i, Year)] = filtered_data['SM10'].mean()
                mean_data_St[(j, i, Year)] = filtered_data['ST10'].mean()
                mean_data_At[(j, i, Year)] = filtered_data['Air_Temperature'].mean()
                mean_data_R[(j, i, Year)] = filtered_data['Rain'].mean()
                #mean_data_ref[(j, i, Year)] = filtered_data['ref40'].mean()
    
    Year += 1 

mean_data_SM_df = pd.DataFrame(mean_data_SM.items(), columns=['Date', 'SM10_Mean'])
mean_data_St_df = pd.DataFrame(mean_data_St.items(), columns=['Date', 'St10_Mean'])
mean_data_At_df = pd.DataFrame(mean_data_At.items(), columns=['Date', 'Air_Temperature_Mean'])
mean_data_R_df = pd.DataFrame(mean_data_R.items(), columns=['Date', 'Rain_Mean'])


mean_data_SM_df['Date'] = mean_data_SM_df['Date'].apply(lambda x: pd.Timestamp(year=x[2], month=x[1], day=x[0]))
mean_data_St_df['Date'] = mean_data_St_df['Date'].apply(lambda x: pd.Timestamp(year=x[2], month=x[1], day=x[0]))
mean_data_At_df['Date'] = mean_data_At_df['Date'].apply(lambda x: pd.Timestamp(year=x[2], month=x[1], day=x[0]))
mean_data_R_df['Date'] = mean_data_R_df['Date'].apply(lambda x: pd.Timestamp(year=x[2], month=x[1], day=x[0]))


data_with_means = data.merge(mean_data_SM_df, on='Date', how='left')
data_with_means = data_with_means.merge(mean_data_St_df, on='Date', how='left')
data_with_means = data_with_means.merge(mean_data_At_df, on='Date', how='left')
data_with_means = data_with_means.merge(mean_data_R_df, on='Date', how='left')

data_with_means.to_csv('heatmap.csv', index=False)

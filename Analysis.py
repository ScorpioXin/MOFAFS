import pandas as pd


data_aggregation = pd.DataFrame()

beachmark = 'static6'
algorithm = 'ensga'
scheduling_count = '1'
input_file = 'scheduling'
output_file = 'analysis'
for running_count in range(1, 21):
    file_path = f'FitnessData/{beachmark}/{algorithm}/{input_file}{scheduling_count}_{running_count}.xlsx'
    df = pd.read_excel(file_path)
    last_column = df.iloc[:, -1]
    data = last_column.iloc[:].values
    data_aggregation[f'count{running_count}'] = data

min_values = data_aggregation.min(axis=1)
mean_values = data_aggregation.mean(axis=1)
std_values = data_aggregation.std(axis=1)

data_aggregation['Min'] = min_values
data_aggregation['Mean'] = mean_values
data_aggregation['Std'] = std_values

data_aggregation.to_excel(f'FitnessData/{beachmark}/{algorithm}/{output_file}.xlsx', index=False)

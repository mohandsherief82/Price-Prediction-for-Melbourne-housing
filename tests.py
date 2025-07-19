import pandas as pd

# create a sample DataFrame
df = pd.read_csv('melb_data.csv')

# find the unique values in column 'A'
unique_values = df['Regionname'].unique()

# print the unique values
print(len(unique_values))
print(unique_values)


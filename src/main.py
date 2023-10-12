import pandas as pd

df = pd.read_csv('training_data .csv')
print(df.head())
y = df['FAT4']
print(y.shape)
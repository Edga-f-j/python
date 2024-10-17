import pandas as pd

# import os
# print(os.getcwd())

df = pd.read_csv("happy.csv")

# Mostrar el DataFrame
print(df['Country'])

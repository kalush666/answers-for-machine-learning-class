import pandas as pd
df = pd.read_csv('winequality-red - winequality-red.csv', sep=',')

df.columns = df.columns.str.strip().str.lower()
print(df.columns.tolist())
import numpy as np
import pandas as pd

np.random.seed(1)
np1= np.random.randint(0,100,30).reshape(5,6)
np1

#1
df = pd.DataFrame(np1)
df.columns=['A','B','C','D','E','F']
df.index=['a','b','c','d','e']

result=df["B"]
print(result)

#2
result=df.loc['c']
print(result)

#3
value = df.loc["c",'D']
print(value)

#4
result = df.loc[['c','e'],['B','F']]
print(result)

#5
result = df.loc[['c', 'd', 'e'], ["C", "D"]]
print(result)

#6
print(df)

#7
result = df.loc['b':'d','A':'F']
print(result)

#8
df['G'] = [80, 147, 102, 118, 99]
result = df.loc['a':'e','A':'G']
print(result)

#9
result = df.index.to_series().map(lambda x: df.index.get_loc(x)%2==0)
print(result)

#10
df[df['C'] > 30]
print(df[df['C'] > 30])

#11
result = (df['B'] > 20) & (df['D'] < 80)
print(result)

#12
result = df > 50
print(result)

#13
result = df[df > 50]
print(result)
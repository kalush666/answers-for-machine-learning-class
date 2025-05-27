import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('students.csv')

#1
print(df)

#2
result = df[df['grade'] > 55]
print('\n', result)

#3
result = df[(df['grade'] < 55) & (df['student_age'] > 25)]
print('\n', result)

#4
data = {'age': df['student_age'], 'study hours': df['hours_of_study']}
result = pd.DataFrame(data)
print('\n', result)

#5
filtered = df['hours_of_study'] >= 3
kplot = plt.axes(projection='3d')
kplot.scatter3D(df['student_age'][filtered], df['hours_of_study'][filtered], df['grade'][filtered])
plt.show()

#6
grade_mean = df['grade'].mean()
df['grade_improve'] = df['grade'].apply(lambda x: x if x > grade_mean else grade_mean)

print(df)
import pandas as pd
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, 0].values 
y = dataset.iloc[:, 1].values  

dic = {}

n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
b1 = numerator / denominator
b0 = y_mean - b1 * x_mean

y_pred_linear = b0 + b1 * x

X_poly = np.vstack([x**2, x, np.ones(n)]).T
coeffs = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
a, b, c = coeffs
y_pred_poly = a * x**2 + b * x + c

mse_linear = np.mean((y - y_pred_linear) ** 2)
mse_poly = np.mean((y - y_pred_poly) ** 2)

dic['Linear'] = mse_linear
dic['Polynomial'] = mse_poly

print("best model is", min(dic, key=dic.get), "with MSE:", dic[min(dic, key=dic.get)])

import matplotlib.pyplot as plt
plt.scatter(x, y, color='blue', label='נתונים')
plt.plot(x, y_pred_linear, color='red', label='Linear')
plt.plot(np.sort(x), y_pred_poly[np.argsort(x)], color='green', label='Polynomial')
plt.xlabel('וותק')
plt.ylabel('שכר')
plt.legend()
plt.show()
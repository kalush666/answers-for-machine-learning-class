import pandas as pd
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, 0].values 
y = dataset.iloc[:, 1].values  

dic = {}

n = len(x)
x_mean = np.mean(x)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

def predict(x, coeffs, bias):
    return sum([coeffs[i] * (x ** i) for i in range(len(coeffs))]) + bias

def compute_cost(x, y, coeffs, bias):
    N = len(y)
    total_error = 0
    for i in range(N):
        y_pred = predict(x[i], coeffs, bias)
        total_error += (y_pred - y[i]) ** 2
    return total_error / N

def gradient_descent(x, y, coeffs, bias, learning_rate, iterations):
    N = len(y)
    degree = len(coeffs) - 1

    for i in range(iterations):
        gradients = [0] * len(coeffs)
        bias_gradient = 0

        for j in range(N):
            y_pred = predict(x[j], coeffs, bias)
            error = y_pred - y[j]
            for d in range(degree + 1):
                gradients[d] += (2 / N) * error * (x[j] ** d)
            bias_gradient += (2 / N) * error

        for d in range(degree + 1):
            coeffs[d] -= learning_rate * gradients[d]
        bias -= learning_rate * bias_gradient

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {compute_cost(x, y, coeffs, bias)}, Coefficients: {coeffs}, Bias: {bias}")

    return coeffs, bias

coeffs = [0, 0, 0]  # [a0, a1, a2] for a0 + a1*x + a2*x^2
bias = 0

learning_rate = 0.0001
iterations = 2000

coeffs, bias = gradient_descent(x, y, coeffs, bias, learning_rate, iterations)

y_pred_poly = np.array([predict(val, coeffs, bias) for val in x])

mse_poly = np.mean((y - y_pred_poly) ** 2)
print(f"Polynomial Regression MSE (Gradient Descent): {mse_poly:.2f}")

plt.scatter(x, y, color='blue', label='Data')
plt.plot(np.sort(x), y_pred_poly[np.argsort(x)], color='green', label='Polynomial GD')
plt.xlabel('וותק')
plt.ylabel('שכר')
plt.legend()
plt.show()
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

dif = 0
dif = dic[max(dic, key=dic.get)] - dic[min(dic, key=dic.get)]
print("Difference in MSE:", dif)

import matplotlib.pyplot as plt
plt.scatter(x, y, color='blue', label='נתונים')
plt.plot(x, y_pred_linear, color='red', label='Linear')
plt.plot(np.sort(x), y_pred_poly[np.argsort(x)], color='green', label='Polynomial')
plt.xlabel('וותק')
plt.ylabel('שכר')
plt.legend()
plt.show()
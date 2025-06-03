import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient_descent(f, learning_rate, iterations, x0, show_graph=False, print_df=False):
    history = []
    x = x0

    for i in range(iterations):
        grad = numerical_derivative(f, x)
        x_new = x - learning_rate * grad
        history.append({"iteration": i, "X": x, "f`(x)": grad, "Xnew": x_new})
        if abs(x - x_new) < 1e-10:
            break
        x = x_new

    df = pd.DataFrame(history)
    if print_df:
        print(df.to_string(index=False))

    min_x = df.iloc[-1]["Xnew"]
    min_y = f(min_x)

    if show_graph:
        x_vals = np.linspace(min_x - 5, min_x + 5, 400)
        y_vals = f(x_vals)
        plt.plot(x_vals, y_vals, label='f(x)', color='blue')
        plt.scatter(min_x, min_y, color='red', s=80, label='Minimum')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gradient Descent: Function and Minimum')
        plt.legend()
        plt.grid(True)
        plt.show()

    return min_x
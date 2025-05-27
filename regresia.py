import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('winequality-red - winequality-red.csv')

X = df.drop('quality', axis=1)
y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'mse': mse, 'r2': r2}

for name, res in results.items():
    print(f"{name}: MSE={res['mse']:.3f}, R2={res['r2']:.3f}")

best_model = max(results, key=lambda x: results[x]['r2'])
print(f"\nהמודל הטוב ביותר הוא: {best_model}")

selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support(indices=True)]
print("\nהפיצ'רים שנבחרו:")
print(selected_features)

X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_new, y, test_size=0.25, random_state=42)
final_model = models[best_model]
final_model.fit(X_train_fs, y_train_fs)
y_pred_fs = final_model.predict(X_test_fs)
mse_fs = mean_squared_error(y_test_fs, y_pred_fs)
r2_fs = r2_score(y_test_fs, y_pred_fs)
print(f"\n{best_model} עם בחירת פיצ'רים: MSE={mse_fs:.3f}, R2={r2_fs:.3f}")
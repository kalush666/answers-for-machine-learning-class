import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

preds = {}

# Load and preprocess data
dataset = pd.read_csv('winequality-red - winequality-red.csv')
dataset['quality'] = dataset['quality'] - 3
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svm_params = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
svm_grid = GridSearchCV(SVC(random_state=0), svm_params, cv=3)
svm_grid.fit(X_train, y_train)
y_pred = svm_grid.predict(X_test)
preds['SVM'] = accuracy_score(y_test, y_pred)

dt_params = {'max_depth': [None, 5, 10, 20], 'criterion': ['gini', 'entropy']}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), dt_params, cv=3)
dt_grid.fit(X_train, y_train)
y_pred = dt_grid.predict(X_test)
preds['DecisionTree'] = accuracy_score(y_test, y_pred)

knn_params = {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(metric='minkowski', p=2), knn_params, cv=3)
knn_grid.fit(X_train, y_train)
y_pred = knn_grid.predict(X_test)
preds['KNN'] = accuracy_score(y_test, y_pred)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
preds['NaiveBayes'] = accuracy_score(y_test, y_pred)

maxkey = max(preds, key=preds.get)
print(f"Best model is {maxkey} with accuracy {preds[maxkey]:.2f}")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df=pd.read_csv("Social_Network_Ads.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X = np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sns.scatterplot(x=df['Age'],y=df['EstimatedSalary'],hue=df['Purchased'])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def sigmoid(z):
  return 1/(1+np.exp(-z))

def y_hat(x,w,b):
  return sigmoid(w@x.T+b)

w = np.zeros((2),dtype='float')
b = 0

def J(y,y_hat):
  return np.mean(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat))

y_hat_ = y_hat(X,w,b)
J(y,y_hat_)

def dw(X,y,y_hat):
  return np.mean((y_hat-y) @ X)

def db(y,y_hat):
  return np.mean(y_hat-y)

alpha = 0.01

J_lst=[]

for e in range(1000):
    y_hat_ = y_hat(X_train, w, b)
    w_step = dw(X_train, y_train, y_hat_)
    b_step = db(y_train, y_hat_)
    w -= alpha * w_step
    b -= alpha * b_step

    y_hat_ = y_hat(X_train, w, b)
    J_lst.append(J(y_train, y_hat_))

plt.plot(J_lst)
plt.xlabel("Epoch")
plt.ylabel("Loss (J)")
plt.title("Loss over epochs")
plt.show()

print ('w=',w,'b=',b)
y_perd= y_hat(X_test,w,b)



y_perd_fix = []
for yp in y_perd:
    if yp >= 0.5:
        y_perd_fix.append(1)
    else:
        y_perd_fix.append(0)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_perd_fix)
print(cm)
accuracy_score(y_test, y_perd_fix)
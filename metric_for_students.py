import numpy as np
from sklearn.metgit rics import confusion_matrix, accuracy_score

def metrics(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
    precision = tp/(tp+fp) if (tp+fp)!=0 else 0
    recall = tp/(tp+fn) if (tp+fn)!=0 else 0
    accuracy= (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)!=0 else 0
    f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0 else 0
    return (accuracy,precision,recall,f1_score)

y=[0,0,0,0,1,1,0,0,1,0]
y_hat=[0,0,0,1,1,0,0,1,1,1]
accuracy,percision,recall,f1_score=metrics(y,y_hat)
print ("accuracy is",accuracy,"precision is",percision,"recall is",recall,"f1_score is",f1_score)
cm = confusion_matrix(y, y_hat)
print(cm)
print(accuracy_score(y, y_hat))
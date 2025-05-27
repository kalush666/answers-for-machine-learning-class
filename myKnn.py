import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_dots=[[7,7,-1],[7,4,-1],[3,4,-2],[1,4,-2]]
my_candidate=[[4,10]]

import numpy as np
import matplotlib.pyplot as plt
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.xlim([-1, 11])
plt.ylim([-1, 11])
plt.grid()
bads = np.array([7,7])
goods=np.array([3,1])
y_bads = np.array([7,4])
y_goods= np.array([4,4])
plt.scatter(goods, y_goods, color = "g", marker = "o", s = 40)
plt.scatter(bads, y_bads, color = "r", marker = "o", s = 40)
plt.scatter(my_candidate[0][0], my_candidate[0][1], color = "b", marker = "o", s = 40)

my_sqr_dist = np.zeros((len(my_dots), 2))

def my_dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

for i in range(len(my_dots)):
    my_sqr_dist[i][0] = my_dist(my_candidate[0], my_dots[i])
    my_sqr_dist[i][1] = my_dots[i][2]


sortedArr = my_sqr_dist[my_sqr_dist[:, 0].argsort()]


my_bad=0
my_good=0

for i in range(3):
    if (sortedArr[i][1]==-1):
        my_bad=my_bad+1
    else:
        my_good=my_good+1

if (my_good>my_bad):
    print ("class is class good")
    my_color="g"
else:
    print ("class is class bad")
    my_color="r"


import numpy as np
import matplotlib.pyplot as plt
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.xlim([-1, 11])
plt.ylim([-1, 11])
plt.grid()
bads = np.array([7,7])
goods=np.array([3,1])
y_bads = np.array([7,4])
y_goods= np.array([4,4])
plt.scatter(goods, y_goods, color = "g", marker = "o", s = 40)
plt.scatter(bads, y_bads, color = "r", marker = "o", s = 40)
plt.scatter(my_candidate[0][0], my_candidate[0][1], color = my_color, marker = "o", s = 40)
plt.show()


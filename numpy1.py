import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

#1
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], ndmin=6)
print('\n', arr.ndim)

#2
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print('\n', arr[-1, -1])

#3
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print('\n', arr[1:5])

#4
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print('\n', arr[:4])

#5
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print('\n', arr)
print('\nShape of array:', arr.shape)
print('\n', arr[0][1], '\n', arr[0][-1])

#6
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8,9])
newarr = arr.reshape(3, 3)
print('\n', newarr)

#7
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for num in np.nditer(arr):
    print(num)

#8
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print('\n', x)

#9
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr % 2 == 0)
print('\n', x)

#10
arr = np.array([3, 2, 0, 1])
print('\n', np.sort(arr))

#11
arr = np.array([41, 42, 43, 44])

filter_arr = arr > 42

newarr = arr[2:]

print('\n',filter_arr)
print('\n',newarr)

#12
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print (arr)


print (arr.shape)

print (arr.shape[0])
print (arr.shape[1])

newarr = arr.reshape(4, 2)

print('\n',newarr)

#13
arr=[["avi",90],["benny",80]]

np_arr=np.array(arr)

columnIndex = 1
sortedArr = np_arr[np_arr[:,columnIndex].argsort()]

print (np_arr)

print ("--------------")

print (sortedArr)

#14
arr = np.array([96, 24, 55, 99, 45, 78, 84, 83,79,98])

sorted_arr= np.sort(arr)

print ("two best scores are ",sorted_arr[-1],sorted_arr[-2])

print ("two worst scores are ",sorted_arr[0],sorted_arr[1])

print (arr.shape)
       
#15
a = np.array(42)
x=np.array([[1,2],[1,1]])
y=np.array([2,2])
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
result = a * b
print('\n',result)

#16
x = np.array([[1, 2, 3], [1, 2, 3]])
y = np.array([10, 7, 6])
result = x * y
print('\n', result)

#17
x=np.array([2,1])
y=np.array([[1,1],[1,1],[1,1]])

xt = x.reshape(-1, 1)

print (x)
print (xt)
print(y)
print(y.transpose())
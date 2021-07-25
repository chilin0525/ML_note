# basic

## introduction

NumPy is a Python library used for working with arrays

## import

NumPy is usually imported under the np alias:

```python
import numpy as np
```

checking Numpy version:

```python
import numpy as np
print(np.__version__)
```

## array

### creating 

在 Numpy 裡 array object 稱為 **ndarray**

```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(type(arr))
```

output:

```
[1 2 3 4 5]
<class 'numpy.ndarray'>
```

```array()``` parameter 可以是 a list, tuple or any array-like object, and it will converted into an ```ndarray```

```python
import numpy as np

arr = np.array((1, 2, 3, 4, 5))
print(arr)
```

output:

```
[1 2 3 4 5]
<class 'numpy.ndarray'>
```

```array()``` with multiple arguments will raise error:

```python
import numpy as np

arr = np.array(1, 2, 3, 4, 5)
print(arr)
```

output:

```
Traceback (most recent call last):
File "test.py", line 3, in <module>
    arr = np.array(1, 2, 3, 4, 5)
TypeError: array() takes from 1 to 2 positional arguments but 5 were given
```

### dimensions 

在 Numpy 裡面 dimensions 稱為 **axes**

```python
import numpy as np

# 0-D array
arr0 = np.array(42)
# 1-D array
arr1 = np.array([1, 2, 3, 4, 5])
# 2-D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
# 3-D array
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr0)
print(arr1)
print(arr2)
print(arr3)
```

output:

```
42
[1 2 3 4 5]
[[1 2 3]
 [4 5 6]]
[[[1 2 3]
  [4 5 6]]

 [[1 2 3]
  [4 5 6]]]
```

checking the dimensions:

```
import numpy as np

arr0 = np.array(42)
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr0.ndim)
print(arr1.ndim)
print(arr2.ndim)
print(arr3.ndim)
```

### indexing

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr1[0])
print(arr2[0])
print(arr2[0, 1])
print(arr3[0, 1])
print(arr3[0, 1, 2])
```

output:

```
1
[1 2 3]
2
[4 5 6]
6
```

negative index:

```python
import numpy as np

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print('Last element from 2nd dim: ', arr[1, -1])
```

output:

```
Last element from 2nd dim:  10
```
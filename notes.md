# Here's all my notes

## Numpy

`import numpy as np`

`ndarray`

all array items must have same type

### Scalars

Create NumPy array:
`s =  np.array(5)`

See shape of array:
`s.shape`

### Vectors

Create vector: 
`v = np.array([1,2,3])`

Access element of vector: 
`x = v[1]`

supports advanced indexing (add link here):
`v[1:]`

### Matrices

Create matrix:
`m = np.array([[1,2,3], [4,5,6], [7,8,9]])`

### Tensors

Create tensor: 
`t = np.array([[[[1],[2]],[[3],[4]],[[5],[6]]],[[[7],[8]],\
    [[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]],[[17],[17]]]])`

### Changing Shape

Assume:
`v = np.array([1,2,3,4])`

Use Reshape: 
`x = v.reshape(1,4)`
Is the same as: 
`x = v[None, :]`

### Element operations

Add:
```
values = [1,2,3,4,5]
values = np.array(values) + 5
```
If already an `ndarray`: 
```
values += 5
```

### Matrix Product

`matmul`

```
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])


# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])


c = np.matmul(a, b)
# array([[ 70,  80,  90],
#        [158, 184, 210]])
```

`dot` and `matmul` are the same if the matrices are two dimensional

### Transpose

Use: `.T`

```
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

m.T
# array([[ 1,  5,  9],
#        [ 2,  6, 10],
#        [ 3,  7, 11],
#        [ 4,  8, 12]])
```

**Transpose will modify original matrix**

## Linear Boundaries

### Boundary Lines

w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b = 0

Wx + b = 0 where W=(w<sub>1</sub>,w<sub>2</sub>) and x = (x<sub>1</sub>,x<sub>2</sub>)

Prediction

ŷ = { 1 if Wx + b >= 0 or 0 if Wx +b < 0 }

### Boundary Planes

w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + w<sub>3</sub>x<sub>3</sub> + b = 0

Wx + b = 0

ŷ = { 1 if Wx + b >= 0 or 0 if Wx +b < 0 }

### Perceptron

what checks the input agains the model

Inputs (nodes)

Weights (edges)

Bias (node)

Linear equation into step function
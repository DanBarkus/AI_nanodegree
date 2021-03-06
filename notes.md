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

#### And Perceptron

Both inputs must be true

turn truth table into 1 and 0 table

### Linear EQ

to move line closer to point

3x<sub>1</sub> + 4x<sub>2</sub> -10 = 0

point is (4,5)

subtract point from constants of line eq

3 | 4 | -10

4 | 5 | 1

-1| -1| -11

Learning rate is multiplier for outcome, result subtracted from line

if prediction is 0

add alpha

if prediction is 1 

subtract alpha

### Continuous Predictions

sigmoid(x) = 1/(1+e<sup>-x</sup>)
```
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
```

#### Softmax Fucntion

e<sup>Z<sub>i</sub></sup>/(e<sup>Z<sub>1</sub></sup>+e<sup>Z<sub>2</sub></sup>+e<sup>Z<sub>n</sub></sup>)

#### Probability

Probability = outcome_prob<sub>1</sub> * outcome_prob<sub>2</sub> * outcome_prob<sub>n</sub>

prediction formula
```
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)
```

#### Cross Entropy

-ln(e<sub>1</sub>)-ln(e<sub>2</sub>)-ln(e<sub>3</sub>)

y = outcome (1 or 0)

ŷ = probability (0 to 1)

#### Error Formula / Logistic Regression

e = error (-ln(ŷ) if y=1 or -lm(1-ŷ) if y = 0)

```
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)
```

cross entropy = y<sub>i</sub>ln(p<sub>i</sub>)+(1-y<sub>i</sub>)ln(1-p<sub>i</sub>)

lower number is better

#### Weight and Bias update formula

$E(W,b) = - \frac{1}{m} \sum_{i=1}^{m} (1-y_i)ln(1-sigmoid(Wx^{(i)}+b)+y_iln(sigmoid(Wx^{(i)}+b))$

```
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = -(y - output)
    weights -= learnrate * d_error * x
    bias -= learnrate * d_error
    return weights, bias
```

### Gradient Descent

$o^{\prime}(x) = o(x)(1-o(x))$

$E =  - \frac{1}{m} \sum_{i=1}^{m} (y_iln(\hat{y_i})+(1-y_i)ln(1-\hat{y_i}))$

$\hat{y_i} = o(Wx^{(i)}+b)$

$VE = -yln(\hat{y})-(1-y)ln(1-\hat{y})$

$\hat{y}(1-\hat{y})*x_j$

$VE = -(y-\hat{y})(x_1,\dots,x_n,1)$

$ErrorTerm = (y - \hat{y})*ActivationFunction$

$WeightUpdate = WeightUpdate + LearningRate * ErrorTerm * Input$

## Feed Forward

$o'(x) = o(x)(1-o(x))$

## Steps

### Forward Pass

$\hat y = f(\sum_i w_i x_i)$

### Calculate Error

$\delta = (y - \hat y) * f'(\sum_i w_i x_i)$

### Update Weights

$Δw_i=Δw_i+δx_i$

## Hidden Layers

dimensions = hidden units X input units

## Regularization

### L1

good for feature selection (1's and 0's)

### L2

Better for training model (varied data)

## Dropout

Turns off nodes to prevent forced learning


## CNN in Keras

`from keras.layers import Conv2D`

`Conv2D(filters, kernel_size, strides, padding, activation='relu', input_shape)`

You must pass the following arguments:

- filters - The number of filters.
- kernel_size - Number specifying both the height and width of the (square) convolution window.

There are some additional, optional arguments that you might like to tune:

- strides - The stride of the convolution. If you don't specify anything, strides is set to 1.
- padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.
- activation - Typically 'relu'. If you don't specify anything, no activation is applied. You are strongly encouraged to add a ReLU activation function to every convolutional layer in your networks.

NOTE: It is possible to represent both kernel_size and strides as either a number or a tuple.

## Pooling in Keras

Used to reduce feature size between layers

`from keras.layers import MaxPooling2D`
`MaxPooling2D(pool_size, strides, padding)`

### Pooling Settings

`pool_size` - Number specifying the height and width of the pooling window.

`strides` - The vertical and horizontal stride. If you don't specify anything, strides will default to pool_size.

`padding` - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.

NOTE: It is possible to represent both pool_size and strides as either a number or a tuple.


## Convolutional Layers

```
# output depth
k_output = 64

# image dimensions
image_width = 10
image_height = 10
color_channels = 3

# convolution filter dimensions
filter_size_width = 5
filter_size_height = 5

# input/image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels], name='inputs')

# weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# apply convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# apply activation function
conv_layer = tf.nn.relu(conv_layer)
```

## Pooling Layers

```
...
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# apply max pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

#GANs

discriminator_loss = scroos_entropy(logits,labels * 0.9)
generator_loss = 
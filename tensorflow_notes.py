import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

# Initializes TensorFlow variables from the graph
init = tf.global_variables_initializer()

# Initializing from within a session
session.run(tf.global_variables_initializer())

# Generates weights from normal distribution
tf.truncated_normal()

n_features = 120
n_labels = 5
# Initialize weights with normal distribution
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

# Initialize biases
biases = tf.Variable(tf.zeros(n_labels))

# return tensor with all zeros
tf.zeros()

# Softmax function Returns all inputs normalized between 0 and 1 and summing up to 1
tf.nn.softmax()

# Cross Entropy
-tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
# Reduce Sum sums numbers in an array
tf.reduce_sum()

# Natural log
tf.log()


with tf.Session() as sess:
    sess.run(init)
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)

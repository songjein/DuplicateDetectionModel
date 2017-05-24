import tensorflow as tf
import numpy as np
import random

tf.set_random_seed(777)  # reproducibility

train1 = np.loadtxt('s1.txt', delimiter=',', dtype=np.float32)
train2 = np.loadtxt('s2.txt', delimiter=',', dtype=np.float32)
target = np.loadtxt('s3.txt', delimiter=',', dtype=np.float32)
target = np.reshape(target, (-1, 1))
learning_rate = 0.001

x_data1 = tf.placeholder(tf.float32, [None, 10])
x_data2 = tf.placeholder(tf.float32, [None, 10])
y_data = tf.placeholder(tf.float32, [None, 1])

# weights & bias for nn layers
W1a = tf.Variable(tf.random_normal([10, 1]))
b1a = tf.Variable(tf.random_normal([1]))
L1a = tf.nn.sigmoid(tf.matmul(x_data1, W1a) + b1a)

# W2a = tf.Variable(tf.random_normal([10, 1]))
# b2a = tf.Variable(tf.random_normal([1]))
# L2a = tf.nn.sigmoid(tf.matmul(L1a, W2a) + b2a)

W1b = tf.Variable(tf.random_normal([10, 1]))
b1b = tf.Variable(tf.random_normal([1]))
L1b = tf.nn.sigmoid(tf.matmul(x_data2, W1b) + b1b)

# W2b = tf.Variable(tf.random_normal([10, 1]))
# b2b = tf.Variable(tf.random_normal([1]))
# L2b = tf.nn.sigmoid(tf.matmul(L1b, W2b) + b2b)

x_merged = tf.concat([L1a, L1b], 1)
W3 = tf.Variable(tf.random_normal([2, 1]))
b3 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(x_merged, W3) + b3)

# cost/loss function
cost = -tf.reduce_mean(y_data * tf.log(hypothesis) + (1 - y_data) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        cost_val, _ = sess.run([cost, train], feed_dict={x_data1: train1, x_data2: train2, y_data: target})
        if step % 1000 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={x_data1: train1, x_data2: train2, y_data: target})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

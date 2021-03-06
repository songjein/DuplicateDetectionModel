import tensorflow as tf
import numpy as np
import random

tf.set_random_seed(777)  # reproducibility

train1 = np.loadtxt('../proto/s1.csv', delimiter=',', dtype=np.float32)
train2 = np.loadtxt('../proto/s2.csv', delimiter=',', dtype=np.float32)
target = np.loadtxt('../proto/s3.csv', delimiter=',', dtype=np.float32)
target = np.reshape(target, (-1, 1))


test1 = np.loadtxt('../proto/ts11.txt', delimiter=',', dtype=np.float32)
test2 = np.loadtxt('../proto/ts22.txt', delimiter=',', dtype=np.float32)
test_target = np.loadtxt('../proto/ts33.txt', delimiter=',', dtype=np.float32)
test_target = np.reshape(target, (-1, 1))

learning_rate = 0.001

x_data1 = tf.placeholder(tf.float32, [None, 10])
x_data2 = tf.placeholder(tf.float32, [None, 10])
y_data = tf.placeholder(tf.float32, [None, 1])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([10, 1]))
b1 = tf.Variable(tf.random_normal([1]))
L1 = tf.nn.sigmoid(tf.matmul(x_data1, W1) + b1)
print ("L1")
print (L1)

W2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.random_normal([1]))
L2 = tf.nn.sigmoid(tf.matmul(x_data2, W2) + b2)
print ("L2")
print (L2)

x_merged = tf.concat([L1, L2], 1)
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
                       feed_dict={x_data1: test1, x_data2: test2, y_data: test_target})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


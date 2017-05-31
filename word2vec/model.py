from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random
import numpy as np
import tensorflow as tf

###############################################################################################
# data load
###############################################################################################
def load_data(filename):

	xy = np.load(filename)

	_x1 = xy['train1'].reshape(-1, 50)
	_x2 = xy['train2'].reshape(-1, 50)
	_y = xy['label'].reshape(-1, 1)

	print (_x1.shape,_y.shape)

	shuffle_idx = np.arange(_x1.shape[0])
	np.random.shuffle(shuffle_idx)

	_x1 = _x1[shuffle_idx]
	_x2 = _x2[shuffle_idx]
	_y = _y[shuffle_idx]

	return _x1, _x2, _y

filename = 'train_idx.npz'
_x_data1, _x_data2, _y_data = load_data(filename)
idx_test1 = idx_test2 = []

###############################################################################################
# variables
###############################################################################################
vocabulary_size = 50000
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

w2v_graph = tf.Graph()
fc_graph = tf.Graph()

###############################################################################################
# word embedding
###############################################################################################
with w2v_graph.as_default():
	# Input data.
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	# Ops and variables pinned to the CPU because of missing GPU implementation
	with tf.device('/cpu:0'):
		# Look up embeddings for inputs.
		embeddings = tf.Variable(
			tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)

		# Construct the variables for the NCE loss
		nce_weights = tf.Variable(
			tf.truncated_normal([vocabulary_size, embedding_size],
								stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each
	# time we evaluate the loss.
	loss = tf.reduce_mean(
		tf.nn.nce_loss(weights=nce_weights,
					   biases=nce_biases,
					   labels=train_labels,
					   inputs=embed,
					   num_sampled=num_sampled,
					   num_classes=vocabulary_size))

	# Construct the SGD optimizer using a learning rate of 1.0.
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	# Compute the cosine similarity between minibatch examples and all embeddings.
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(
		normalized_embeddings, valid_dataset)
	similarity = tf.matmul(
		valid_embeddings, normalized_embeddings, transpose_b=True)

###############################################################################################
# full connected 
###############################################################################################
with fc_graph.as_default():

	MAX_WORD_LENGTH = 50

	x_idx1 = tf.placeholder(tf.int32, [None, MAX_WORD_LENGTH])
	x_idx2 = tf.placeholder(tf.int32, [None, MAX_WORD_LENGTH])

	x_data1 = tf.placeholder(tf.int32, [None, MAX_WORD_LENGTH, embedding_size])
	x_data2 = tf.placeholder(tf.int32, [None, MAX_WORD_LENGTH, embedding_size])
	y_data = tf.placeholder(tf.float32, [None, 1])
	embeddings = tf.placeholder(tf.float32, [vocabulary_size, embedding_size])

	x_data1 = tf.nn.embedding_lookup(embeddings, x_idx1)
	x_data2 = tf.nn.embedding_lookup(embeddings, x_idx2)

	# weights & bias for nn layers
	W1 = tf.Variable(tf.random_normal([MAX_WORD_LENGTH, embedding_size]))
	b1 = tf.Variable(tf.random_normal([MAX_WORD_LENGTH]))
	# L1 = tf.nn.sigmoid(tf.multiply(x_data1, W1) + b1)
	L1 = tf.nn.tanh(tf.reduce_sum(tf.multiply(x_data1, W1), 2) + b1)
	# ? x 50 x 128 => ? x 50
	print("L1")
	print(L1)

	# first sentence ? x 50 => ? x 10
	W1_1 = tf.Variable(tf.random_normal([MAX_WORD_LENGTH, 10]))
	b1_1 = tf.Variable(tf.random_normal([10]))
	L1_1 = tf.nn.tanh(tf.matmul(L1, W1_1) + b1_1)
	print("L1_1")
	print(L1_1)

	W2 = tf.Variable(tf.random_normal([MAX_WORD_LENGTH, embedding_size]))
	b2 = tf.Variable(tf.random_normal([MAX_WORD_LENGTH]))
	# L2 = tf.nn.sigmoid(tf.multiply(x_data2, W2) + b2)
	L2 = tf.nn.tanh(tf.reduce_sum(tf.multiply(x_data2, W2), 2) + b2)
	# ? x 50 x 128  => ? x 50
	print("L2")
	print(L2)

	# second sentence ? x 50 => ? x 10
	W2_1 = tf.Variable(tf.random_normal([MAX_WORD_LENGTH, 10]))
	b2_1 = tf.Variable(tf.random_normal([10]))
	L2_1 = tf.nn.tanh(tf.matmul(L2, W2_1) + b2_1)
	print("L2_1")
	print(L2_1)

	x_merged = tf.concat([L1_1, L2_1], 1)  # (MAX_WORD_LENGTH * 2) * 1
	# ? x 20
	print("x_merged")
	print(x_merged)

	W3 = tf.Variable(tf.random_normal([2 * 10, 1]))  # each sentence => 1 output
	b3 = tf.Variable(tf.random_normal([1]))

	hypothesis = tf.sigmoid(tf.matmul(x_merged, W3) + b3)
	print("hypothesis")
	print(hypothesis)

	# cost/loss function
	cost = -tf.reduce_mean(y_data * tf.log(hypothesis) + (1 - y_data) * tf.log(1 - hypothesis))

	train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

	# Accuracy computation
	# True if hypothesis>0.5 else False
	predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), dtype=tf.float32))

with tf.Session(graph=w2v_graph) as session:
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	# We must initialize all variables before we use them.
	init.run()
	print('Initialized')

	# save the variables to disk
	save_path = saver.restore(session, "./saver/w2v.ckpt")
	print("model restored")

	# final_embeddings, dictionary
	final_embeddings = normalized_embeddings.eval()

iteration = 200
batch_num = 10000
###############################################################################################
# for initializing the word embeddings from checkpoint!!
###############################################################################################
with tf.Session(graph=fc_graph) as session:
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	init.run()

	train_size = int(len(_y_data) * 0.9)

	for epoch in range(iteration):

		avg_cost = 0
		total_batch = int(train_size / batch_num)

		for iter in range(total_batch):
			batch_x1 = _x_data1[iter * batch_num : (iter + 1) * batch_num]
			batch_x2 = _x_data2[iter * batch_num : (iter + 1) * batch_num]
			batch_y  = _y_data[iter * batch_num: (iter + 1) * batch_num]

			c, _, = session.run([cost, train], feed_dict={x_idx1: batch_x1, x_idx2: batch_x2, y_data: batch_y, embeddings: final_embeddings})
			avg_cost += c / total_batch

		save_path = saver.save(session, "./save_model/%dmodel.ckpt" % (epoch))
		print("model saved in file: %s" % save_path)
		print('Epoch:', '%04d' % epoch, 'cost =', avg_cost)

		# Accuracy report
		a = session.run(accuracy, feed_dict={x_idx1: _x_data1[train_size:], x_idx2: _x_data2[train_size:], y_data: _y_data[train_size:], embeddings: final_embeddings})
		print("\nAccuracy: ", a)
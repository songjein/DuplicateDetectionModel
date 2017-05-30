from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import csv

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# Read the data into a list of strings.
def make_voca(filename):
	data = []
	with open(filename, 'r', encoding='utf-8') as f:
	# with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for line in reader:
			data += line[3].split() + line[4].split()
	return data

filename = './train.csv'
vocabulary = make_voca(filename)

print('Data size', len(vocabulary))

# # Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

# make data=>index, dictionary, reverse index....
def build_dataset(words, n_words):
	"""Process raw inputs into a dataset."""
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(n_words - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	# data: input data => index , count => each word count, dictionary => word:index, reversed_dictionary
	return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


graph = tf.Graph()

with graph.as_default():

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

	# Add variable initializer.
	init = tf.global_variables_initializer()

	# Add ops to save and restore all the variables
	saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
	# We must initialize all variables before we use them.
	init.run()
	print('Initialized')

	# save the variables to disk
	save_path = saver.restore(session, "./saver/w2v.ckpt")
	print ("model restored")

	final_embeddings = normalized_embeddings.eval()

	print ("\n")
	print ("Embedding size : %s x %s" % (len(final_embeddings),len(final_embeddings[0])))
	print ("Our embedding looks like..")
	print (final_embeddings)

	print ("\n")
	print ("Our dictonary looks like..")
	print (str(dictionary)[0:500])

	print ("\n")
	print ("Our reversed dictonary looks like..")
	print (str(reverse_dictionary)[0:500])


	while True:
		sentence = raw_input("type sentence : ")
		words = sentence.split()
		print (words)

		w2v = []
		for w in words:
			w2v.append(final_embeddings[dictionary[w]])

		print (w2v)


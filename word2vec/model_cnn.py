from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model_cnn_class import TextCNN
import math
import numpy as np
import tensorflow as tf

tf.set_random_seed(777)  # reproducibility

###############################################################################################
# data load
###############################################################################################
def load_data(filename):
    xy = np.load(filename)

    _x1 = xy['train1'].reshape(-1, 50)
    _x2 = xy['train2'].reshape(-1, 50)
    _y = xy['label'].reshape(-1, 1)

    print(_x1.shape, _y.shape)

    shuffle_idx = np.arange(_x1.shape[0])
    np.random.shuffle(shuffle_idx)

    _x1 = _x1[shuffle_idx]
    _x2 = _x2[shuffle_idx]
    _y = _y[shuffle_idx]

    return _x1, _x2, _y


# (404290, 50, 1) (404290, 50, 1) (404290,)
filename = 'train_idx.npz'

_x_data1, _x_data2, _y_data = load_data(filename)
idx_test1 = []
idx_test2 = []

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
    with tf.device('/gpu:0'):
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
# for initializing the word embeddings from checkpoint!!
###############################################################################################
with tf.Session(graph=w2v_graph) as session:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    # save the variables to disk
    save_path = saver.restore(session, "./save_w2v/w2v-50000.ckpt")
    print("word2vec restored")

    # final_embeddings, dictionary
    final_embeddings = normalized_embeddings.eval()

###############################################################################################
# training
###############################################################################################
# Parameters
MAX_WORD_LENGTH = 50
filter_sizes    = [2, 4]
num_filters     = 20
num_epochs      = 64
batch_num       = 128

with tf.Session(graph=fc_graph) as session:

    cnn = TextCNN(MAX_WORD_LENGTH, vocabulary_size, embedding_size, filter_sizes, num_filters)

    # Define training procedure
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(cnn.cost)

    train_size = int(len(_y_data) * 0.9)
    test_size = len(_y_data) - train_size

    trainX1, testX1 = np.array(_x_data1[0:train_size]), np.array(_x_data1[train_size:len(_y_data)])
    trainX2, testX2 = np.array(_x_data2[0:train_size]), np.array(_x_data2[train_size:len(_y_data)])
    trainY, testY = np.array(_y_data[0:train_size]), np.array(_y_data[train_size:len(_y_data)])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    init.run()

    for epoch in range(num_epochs):

        avg_cost = 0
        total_batch = int(train_size / batch_num)

        for iteration in range(total_batch):

            start_index = iteration * batch_size
            end_index = min((iteration + 1) * batch_size, train_size)
            batch_x1 = trainX1[start_index:end_index]
            batch_x2 = trainX2[start_index:end_index]
            batch_y = trainY[start_index:end_index]

            # A single training step
            _, loss = session.run([train_op, cnn.cost],
                                  feed_dict={
                                      cnn.x_idx1: batch_x1,
                                      cnn.x_idx2: batch_x2,
                                      cnn.y_data: batch_y,
                                      cnn.embeddings: final_embeddings,
                                      cnn.dropout_keep_prob: 0.7 })
            avg_cost += loss / total_batch

        save_path = saver.save(session, "./save_model/cnn%dmodel.ckpt" % (epoch))

        print('Epoch:', '%04d' % epoch, 'cost =', avg_cost)
        # Accuracy report
        a = session.run(cnn.accuracy,
                        feed_dict={
                            cnn.x_idx1: testX1[train_size:],
                            cnn.x_idx2: testX2[train_size:],
                            cnn.y_data: testY[train_size:],
                            cnn.embeddings: final_embeddings,
                            cnn.dropout_keep_prob: 1.0})
        print("Accuracy: ", a)

    # Accuracy report
    a = session.run(cnn.accuracy,
                    feed_dict={
                        cnn.x_idx1: testX1[train_size:],
                        cnn.x_idx2: testX2[train_size:],
                        cnn.y_data: testY[train_size:],
                        cnn.embeddings: final_embeddings,
                        cnn.dropout_keep_prob: 1.0})
    print("Accuracy: ", a)
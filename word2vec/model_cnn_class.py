import tensorflow as tf

class TextCNN(object):
    """
    A CNN for text classification.
    """
    def __init__(
      self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters):

        # Placeholders for input, output and dropout
        self.x_idx1 = tf.placeholder(tf.int32, [None, sequence_length], name='idx1')
        self.x_idx2 = tf.placeholder(tf.int32, [None, sequence_length], name='idx2')
        print (self.x_idx1)
        print (self.x_idx2)

        self.x_data1 = tf.placeholder(tf.int32, [None, sequence_length, embedding_size, 1], name='x_data1')
        self.x_data2 = tf.placeholder(tf.int32, [None, sequence_length, embedding_size, 1], name='x_data2')
        self.y_data = tf.placeholder(tf.float32, [None, 1], name='y')
        print (self.x_data1)
        print (self.x_data2)
        print (self.y_data)

        self.embeddings = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        print (self.embeddings)

        self.x_data1 = tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.x_idx1), [-1, sequence_length, embedding_size, 1])
        self.x_data2 = tf.reshape(tf.nn.embedding_lookup(self.embeddings, self.x_idx2), [-1, sequence_length, embedding_size, 1])
        print (self.x_data1)
        print (self.x_data2)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs1 = []
        pooled_outputs2 = []

        for i, filter_size in enumerate(filter_sizes):

            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer1
                conv1 = tf.layers.conv2d(
                    inputs=self.x_data1,
                    filters=num_filters,
                    kernel_size=[filter_size, embedding_size],
                    padding='VALID',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                print ('conv1', conv1)

                pooled1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=[sequence_length - filter_size + 1, 1],
                    strides=1,
                    padding='VALID')
                pooled1 = tf.nn.dropout(pooled1, self.dropout_keep_prob)
                print ('pool1', pooled1)

                # Convolution Layer2
                conv2 = tf.layers.conv2d(
                    inputs=self.x_data2,
                    filters=num_filters,
                    kernel_size=[filter_size, embedding_size],
                    padding='VALID',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                print ('conv2', conv2)

                pooled2 = tf.layers.max_pooling2d(
                    inputs=conv2,
                    pool_size=[sequence_length - filter_size + 1, 1],
                    strides=1,
                    padding='VALID')
                pooled2 = tf.nn.dropout(pooled2, self.dropout_keep_prob)
                print ('pool2', pooled2)

                pooled_outputs1.append(pooled1)
                pooled_outputs2.append(pooled2)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool1 = tf.concat(pooled_outputs1, 3)
        self.h_pool2 = tf.concat(pooled_outputs2, 3)

        self.h_pool_flat1 = tf.reshape(self.h_pool1, [-1, num_filters_total])
        self.h_pool_flat2 = tf.reshape(self.h_pool2, [-1, num_filters_total])
        print ('h_pool_flat1', self.h_pool_flat1)
        print ('h_pool_flat2', self.h_pool_flat2)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop1 = tf.nn.dropout(self.h_pool_flat1, self.dropout_keep_prob)
            self.h_drop2 = tf.nn.dropout(self.h_pool_flat2, self.dropout_keep_prob)

        # Merged fully connected layer1
        with tf.name_scope("merged_layer"):
            self.fc_merged1      = tf.concat([self.h_drop1, self.h_drop2], 1)
            self.fc_merged_drop1 = tf.nn.dropout(self.fc_merged1, self.dropout_keep_prob)
        print ('fc_merged1', self.fc_merged_drop1)

        # Merged fully connected layer2
        with tf.name_scope("merged_layer2"):
            self.fc_merged2      = tf.contrib.layers.fully_connected(self.fc_merged_drop1, 32, activation_fn=tf.nn.relu)
            self.fc_merged_drop2 = tf.nn.dropout(self.fc_merged2, self.dropout_keep_prob)
        print('fc_merged2', self.fc_merged_drop2)

        # Final outputs and predictions
        with tf.name_scope("output"):
            self.hypothesis = tf.contrib.layers.fully_connected(self.fc_merged_drop2, 1, activation_fn=tf.nn.sigmoid)
            print('hypothesis', self.hypothesis)

            self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
            print('predicted', self.predicted)

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.cost = -tf.reduce_mean(self.y_data * tf.log(self.hypothesis) + (1 - self.y_data) * tf.log(1 - self.hypothesis))

        # Accuracy
        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.y_data), dtype=tf.float32))

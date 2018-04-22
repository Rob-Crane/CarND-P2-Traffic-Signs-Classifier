import tensorflow as tf
import numpy as np
from collections import deque
import datetime
from signs_input import get_train_batch, get_validation_batch

def get_network(n_classes, l2_scale, hidden_units):
    
    def inference(images):

        regularizer = tf.contrib.layers.l2_regularizer(
                scale = l2_scale)

        with tf.variable_scope('blk1', reuse = tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(
                    inputs=images,
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv1',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=64,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv2',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            pool = tf.layers.max_pooling2d(
                    inputs=conv2,
                    pool_size=[2,2],
                    strides=2,
                    name='pool')

        with tf.variable_scope('blk2', reuse = tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(
                    inputs=pool,
                    filters=128,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv1',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=128,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv2',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=128,
                    kernel_size=1,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv3',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            pool = tf.layers.max_pooling2d(
                    inputs=conv3,
                    pool_size=[2,2],
                    strides=2,
                    name='pool')

        with tf.variable_scope('blk3', reuse = tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(
                    inputs=pool,
                    filters=256,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv1',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=256,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv2',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=256,
                    kernel_size=1,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv3',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            pool = tf.layers.max_pooling2d(
                    inputs=conv3,
                    pool_size=[2,2],
                    strides=2,
                    name='pool')

        with tf.variable_scope('blk4', reuse = tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(
                    inputs=pool,
                    filters=512,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv1',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=512,
                    kernel_size=3,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv2',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=512,
                    kernel_size=1,
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv3',
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)

        with tf.variable_scope('fc', reuse = tf.AUTO_REUSE):
            area_dim = conv3.get_shape()[1]
            pooled_col = tf.layers.average_pooling2d(
                    conv3,
                    pool_size=(area_dim, area_dim),  # pool entirety of area
                    strides=1,
                    name='pool')

            net = tf.reshape(
                    pooled_col,
                    [-1, np.prod(pooled_col.get_shape()[1:])],
                    name='flat')

            for units in hidden_units:
                net = tf.layers.dense(
                        net,
                        units=units,
                        activation=tf.nn.relu,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer)
                net = tf.layers.dropout(
                        net)

            logits = tf.layers.dense(
                    net,
                    n_classes,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)

        return logits

    return inference

def get_loss(logits, labels):

    xent = tf.losses.sparse_softmax_cross_entropy(
            labels = labels,
            logits = logits)

    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = xent + tf.losses.get_regularization_loss() 

    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predictions,
                               name='acc_op')[1]
    return (loss, accuracy)


LEARN_RATE = 0.1
L2_SCALE = 0.0005
HIDDEN_UNITS = [256,256]
TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 256
MAX_STEPS = 500000
TRAIN_LOG_INTVAL = 200
VALIDATION_LOG_INTVAL = 1000
THRESHOLD_BUFFER = 5

(train_images, train_labels) = get_train_batch(batch_size=TRAIN_BATCH_SIZE)

# create the network with specified hidden layers
inference = get_network(10, L2_SCALE, HIDDEN_UNITS)

train_logits = inference(train_images)

(train_loss, train_accuracy) = get_loss(train_logits, train_labels)

tf.add_to_collection('train_summaries',
        tf.summary.scalar('Training Loss', train_loss ))
tf.add_to_collection('train_summaries',
        tf.summary.scalar('Training Accuracy', train_accuracy))

train_op = tf.train.AdadeltaOptimizer(
        LEARN_RATE).minimize(train_loss)

# Compute evaluation metrics.
(test_images, test_labels) = get_validation_batch(batch_size = VALIDATION_BATCH_SIZE)
test_logits = inference(test_images)

(test_loss, test_accuracy) = get_loss(test_logits, test_labels)

tf.add_to_collection('test_summaries',
        tf.summary.scalar('Validation Loss', test_loss))
tf.add_to_collection('test_summaries',
        tf.summary.scalar('Validation Accuracy', test_accuracy))

loss_delta = test_loss - train_loss
tf.add_to_collection('test_summaries',
        tf.summary.scalar('Loss Delta', loss_delta))

acc_delta = train_accuracy - test_accuracy
tf.add_to_collection('test_summaries',
        tf.summary.scalar('Acc Delta',  acc_delta))

saver = tf.train.Saver()

with tf.Session() as sess:
    timestamp = str(datetime.datetime.now())
    writer = tf.summary.FileWriter('logdir/' + 
            timestamp,
            sess.graph)
    sess.run((tf.global_variables_initializer(),
        tf.local_variables_initializer()))

    for i in range(MAX_STEPS):

        fetches = [train_op]
        train_logging = i % TRAIN_LOG_INTVAL == 0

        if train_logging:
            fetches.append(tf.get_collection('train_summaries'))

        results = sess.run(fetches)

        if train_logging:
            for summary in results[1]:
                writer.add_summary(summary, i)

        if i % VALIDATION_LOG_INTVAL == 0:
            fetches = (test_loss, tf.get_collection('test_summaries'))
            results = sess.run(fetches)

            for test_summary in results[1]:
                writer.add_summary(test_summary,i)

    saver.save(sess, 'models/' + timestamp + '.ckpt')

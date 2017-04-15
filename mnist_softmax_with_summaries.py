# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  with tf.name_scope('weights'):
    W = tf.Variable(tf.zeros([784, 10]))
    variable_summaries(W)
  with tf.name_scope('biases'):
    b = tf.Variable(tf.zeros([10]))
    variable_summaries(b)
  with tf.name_scope('Wx_plus_b'):
    y = tf.matmul(x, W) + b
    tf.summary.histogram('pre_activations', y)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  sess = tf.InteractiveSession()
  train_writer = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/mnist_softmax_with_summaries' + '/train' ,sess.graph)
  test_writer = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/mnist_softmax_with_summaries' + '/test')
  tf.global_variables_initializer().run()

  # Train
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 10 == 0:
      summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
      train_writer.add_summary(summary, i)
  train_writer.close()
  train_writer.flush()
  test_writer.close()
  test_writer.flush()

  # Test trained model
  #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      #y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.summaries_dir = '~/tensorflow/softmaxtest'
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

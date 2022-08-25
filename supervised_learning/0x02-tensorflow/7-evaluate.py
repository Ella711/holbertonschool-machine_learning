#!/usr/bin/env python3
"""
Build, train and save function
"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """ Evaluates the output of a neural network """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        # load params
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        eval_dict = {
            x: X,
            y: Y
        }

        y_eval = sess.run(y_pred, feed_dict=eval_dict)
        eval_acc = sess.run(accuracy, feed_dict=eval_dict)
        eval_loss = sess.run(loss, feed_dict=eval_dict)
        return y_eval, eval_acc, eval_loss

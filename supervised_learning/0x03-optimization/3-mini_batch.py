#!/usr/bin/env python3
"""
3. Mini-Batch
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent
    X_train: ndarray - (m, 784) containing the training data
    Y_train: one-hot ndarray - (m, 10) containing training labels
    X_valid: ndarray - (m, 784) containing the validation data
    Y_valid: one-hot ndarray - (m, 10) containing validation labels
    batch_size: number of data points in a batch
    epochs: iterations through the whole dataset
    load_path: path from which to load the model
    save_path: path to where the model should be saved after training
    Returns: the path where the model was saved
    """
    with tf.Session() as sess:  # import meta graph and restore session
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]  # load tensors and ops
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        feed_train = {x: X_train, y: Y_train}
        feed_valid = {x: X_valid, y: Y_valid}
        m = X_train.shape[0]  # Form batch sizes
        batches = m // batch_size
        if m % batch_size != 0:
            batches += 1
        for i in range(epochs + 1):  # loop over epochs
            train_cost, train_acc = sess.run([loss, accuracy], feed_train)
            valid_cost, valid_acc = sess.run([loss, accuracy], feed_valid)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))
            if i < epochs:
                x_shuffle, y_shuffle = shuffle_data(X_train, Y_train)
                for j in range(batches):
                    start = j * batch_size
                    end = start + batch_size
                    step = j + 1
                    x_batch = x_shuffle[start:end]
                    y_batch = y_shuffle[start:end]
                    feed_batch = {x: x_batch, y: y_batch}
                    sess.run(train_op, feed_batch)  # train
                    if step % 100 == 0:
                        cost, acc = sess.run([loss, accuracy], feed_batch)
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(acc))
        return saver.save(sess, save_path)

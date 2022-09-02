#!/usr/bin/env python3
"""
15. Put it all together and what do you get?
"""
import numpy as np
import tensorflow.compat.v1 as tf


def forward_prop(prev, layers, activations, epsilon):
    """
    Forward prop function with batch norm
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    prev_layer = prev

    for i in range(len(layers) - 1):
        densor = tf.keras.layers.Dense(units=layers[i],
                                       kernel_initializer=init)(prev_layer)

        mean, variance = tf.nn.moments(densor, axes=[0])

        gamma = tf.Variable(tf.ones(layers[i]), trainable=True)
        beta = tf.Variable(tf.zeros(layers[i]), trainable=True)

        batch_norm = tf.nn.batch_normalization(densor,
                                               mean=mean,
                                               variance=variance,
                                               offset=beta,
                                               scale=gamma,
                                               variance_epsilon=epsilon)

        prev_layer = activations[i](batch_norm)

    output_layer = tf.keras.layers.Dense(layers[-1],
                                         activation=None,
                                         kernel_initializer=init)(prev_layer)
    return output_layer


def create_placeholders(nx, classes):
    """ Returns two placeholders, x and y, for the neural network """
    x = tf.placeholder(dtype="float32", shape=(None, nx.shape[1]), name="x")
    y = tf.placeholder(dtype="float32", shape=(None, classes.shape[1]),
                       name="y")
    return x, y


def calculate_accuracy(y, y_pred):
    """ Calculates the accuracy of a prediction """
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """ Calculates the softmax cross-entropy loss of a prediction """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in
        tensorflow using inverse time decay
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    shuffle = np.random.permutation(X.shape[0])
    return X[shuffle, :], Y[shuffle]


def create_train_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network
    in tensorflow using Adam optimization algorithm
    """
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in
        tensorflow using Adam optimization, mini-batch gradient
        descent, learning rate decay, and batch normalization
    """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x, y = create_placeholders(X_train, Y_train)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    decay_steps = 1

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_steps)

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    train_op = create_train_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    store = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        steps = m // batch_size + 1

        for epoch in range(epochs + 1):
            train_accuracy, train_cost = sess.run(
                [accuracy, loss], feed_dict={x: X_train, y: Y_train})
            valid_accuracy, valid_cost = sess.run(
                [accuracy, loss], feed_dict={x: X_valid, y: Y_valid})

            # print training and validation cost and accuracy
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_accuracy))
            print("\tTraining Accuracy: {}".format(train_cost))
            print("\tValidation Cost: {}".format(valid_accuracy))
            print("\tValidation Accuracy: {}".format(valid_cost))

            if epoch == epochs:
                break

            # shuffle data
            x_shuffle, y_shuffle = shuffle_data(X_train, Y_train)

            for step in range(steps):
                start = batch_size * step
                end = batch_size * (step + 1)

                # get X_batch and Y_batch from shuffled trains
                x_batch = x_shuffle[start:end]
                y_batch = y_shuffle[start:end]

                # run training operation
                sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

                # print batch cost and accuracy
                if (step + 1) % 100 == 0:
                    step_accuracy, step_cost = sess.run(
                        [accuracy, loss], feed_dict={x: x_batch, y: y_batch})

                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_accuracy))
                    print("\t\tAccuracy: {}".format(step_cost))

            sess.run(tf.assign(global_step, global_step + 1))

        # save and return the path to where the model was saved
        return store.save(sess, save_path)

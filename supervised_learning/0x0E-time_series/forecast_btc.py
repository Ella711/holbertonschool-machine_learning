#!/usr/bin/env python3
"""
0. When to Invest
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_series(x, y, format="-", start=0, end=None,
                title=None, xlabel=None, ylabel=None, legend=None ):
    """
    Visualizes time series data

    Args:
      x (array of int) - contains values for the x-axis
      y (array of int or tuple of arrays) - contains the values for the y-axis
      format (string) - line style when plotting the graph
      start (int) - first time step to plot
      end (int) - last time step to plot
      title (string) - title of the plot
      xlabel (string) - label for the x-axis
      ylabel (string) - label for the y-axis
      legend (list of strings) - legend for the plot
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))

    # Check if there are more than two series to plot
    if type(y) is tuple:
      # Loop over the y elements
      for y_curr in y:
        # Plot the x and current y values
        plt.plot(x[start:end], y_curr[start:end], format)
    else:
      # Plot the x and y values
      plt.plot(x[start:end], y[start:end], format)

    # Label axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
      plt.legend(legend)
    plt.title(title)
    plt.grid(True)
    plt.show()


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def model_forecast(model, series, window_size, batch_size):
    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get predictions on the entire dataset
    forecast = model.predict(dataset)

    return forecast


if __name__ == "__main__":
    # Initialize lists
    time_step = []
    btc_price = []

    # Open CSV file
    with open('./cleancoinbase.csv') as csvfile:

        # Initialize reader
        reader = csv.reader(csvfile, delimiter=',')

        # Skip the first line
        next(reader)

        # Append row and sunspot number to lists
        for row in reader:
            time_step.append(int(row[0]))
            btc_price.append(float(row[2]))

    # Convert lists to numpy arrays
    time = np.array(time_step)
    series = np.array(btc_price)

    # Preview the data
    plot_series(time, series, xlabel='Hour', ylabel='Hourly Weighted Price of BTC')

    # Define the split time
    # first 3 years
    split_time = 26280

    # Get the train set
    time_train = time[:split_time]
    x_train = series[:split_time]

    # Get the validation set
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # Parameters
    window_size = 24
    batch_size = 32
    shuffle_buffer_size = 2000

    # Generate the dataset windows
    train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    # Build the Model
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(window_size, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # Print the model summary
    model.summary()

    # Get initial weights
    init_weights = model.get_weights()

    # Set the initial learning rate
    initial_learning_rate = 2.8e-06

    # Define the scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=400,
        decay_rate=0.96,
        staircase=True)

    # Set the optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mse"])

    # Train the model
    history = model.fit(train_set, epochs=50)

    # Reduce the original series
    forecast_series = series[split_time - window_size:-1]

    # Use helper function to generate predictions
    forecast = model_forecast(model, forecast_series, window_size, batch_size)

    # Drop single dimensional axis
    results = forecast.squeeze()

    # Plot the results
    plot_series(time_valid, (x_valid, results))

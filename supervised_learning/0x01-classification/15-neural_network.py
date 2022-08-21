#!/usr/bin/env python3
"""
Module for class NeuralNetwork
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Class Constructor

        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1, self.__b2 = np.zeros((nodes, 1)), 0
        self.__A1, self.__A2 = 0, 0
        self.__W2 = np.random.randn(1, nodes)

    @property
    def W1(self):
        """ Getter for W1 """
        return self.__W1

    @property
    def b1(self):
        """ Getter for b1 """
        return self.__b1

    @property
    def A1(self):
        """ Getter for A1 """
        return self.__A1

    @property
    def W2(self):
        """ Getter for W2 """
        return self.__W2

    @property
    def b2(self):
        """ Getter for b2 """
        return self.__b2

    @property
    def A2(self):
        """ Getter for A2 """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = -1 / Y.shape[1]
        cost = m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        A = self.forward_prop(X)[1]
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        m = 1 / X.shape[1]
        dZ2 = A2 - Y
        dW2 = m * np.matmul(dZ2, A1.T)
        db2 = m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = m * np.matmul(dZ1, X.T)
        db1 = m * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the neural network """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        step_array = list(range(0, iterations + 1, step))
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            cost = self.cost(Y, self.__A2)
            if verbose and (i % step == 0 or i == iterations):
                costs.append(cost)
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_array, costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

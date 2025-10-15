import gzip
import pickle
import numpy as np
import os

def load_data():
    """Load MNIST from data/mnist.pkl.gz and return as raw tuples"""
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "mnist.pkl.gz")

    with gzip.open(data_path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    return training_data, validation_data, test_data

def load_data_wrapper():
    """
    Return MNIST data formatted for Network class.
    
    - training_data: list of tuples (x, y) where y is one-hot vector
    - validation_data/test_data: list of tuples (x, y) where y is int
    """
    training_data, validation_data, test_data = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = list(zip(validation_inputs, validation_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return training_data, validation_data, test_data

def vectorized_result(j):
    """Convert digit j into a one-hot 10-dimensional column vector"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


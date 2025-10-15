import random
import numpy as np

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

class Network(object):
    """
        - sizes: contains the numbers of the neurons in the respective layers, for example [2, 3, 1]
        - biases and weights are generated randomly. ! first layer of neurons is an input layer, so oomits to set
        any biasses fot those neurons. 
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #column vector of random numbers, gaussian distributed of shape (y,1). to this for every layer excepts for layer 1. ==> list of vectors.
        self.weights = [np.random.randn(y, x) 
                for x, y in zip(sizes[:-1], sizes[1:])] # sizes[:-1] all layers except the last one, sizes[1:] all the layers except the first one. where (x, y), x: number of neurons;, y: number of neurons in the next layer. (example: [30,10], [10, 1]), first layers 30 nodes, second layer 10 nodes. ==> weights is a list of matrices.

# Notes: biases ans weights are stored as lists of Numpy matrices. So for example: net.weights[1] is a Numpy
# matrix that stores the weight of the second and third layer neurons.


    def feedforward(self, a):
        """Return the output of the network if 'a'a is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
            - training_data: list of tuples, representing the training inputs and corresponding desired outputs.
            - epochs: number of epochs to train for
            - mini_batch_size: size of mini batches to use when trainig.
            - eta: learnign rate
            - test_data: != None => program evaluate the the network after each epoch of training.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        if test_data:
            print("Epoch {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test))
        else:
            print("Epoch {0} complete".format(j+1))
 

    def update_mini_batch(self, mini_batch, eta):
        """
            Update network's weights and biasses by applying gradient descent using backpropagation to a single mini
            batch. The mini_batch is a list of tuples "(x, y)", and "eta" is the learning rate.
        """
        
        # creation of zeros arrays, with same shapes of biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases] # nabla = gradient in math notation
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # loop for each training sample in mini_batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # backprop computes the gradient fo the cost function for that single example. delta_nabla_b and delta_nabla_w contain the weights for each layer.
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # nabla_b = sum of all bias gradients across the mini-batch.
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        """
            Now we actually apply the gradient descent
            w = w - eta*dL/dw
            b = b - eta*dL/db			
        """
        self.weights = [w - ( eta / len(mini_batch) ) * nw
            for w, nw in zip(self.weights, nabla_w)]
        self.biases =  [b - ( eta / len(mini_batch) ) * nb
            for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
            Return a tuple, (nabla_b, nabla_w), representing the gradient for the cost function
            C_x. 'nabla_b' and 'nabla_w' are layer-by-layer lists of numpy arrays, similar
            to 'self.biases' and 'self.weights'.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer, weighted inputs
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # weighted input to the neurons in the current layer
            zs.append(z)
            activation = sigmoid(z) # apply the sigmoid activation function.
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # loops backwards through hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y

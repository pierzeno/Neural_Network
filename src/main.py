import data_loader
import neural_network

def main():
    # Load MNIST data (formatted as (x, y) tuples)
    training_data, validation_data, test_data = data_loader.load_data_wrapper()

    # Create network: 784 inputs (28x28 pixels), 30 hidden neurons, 10 outputs (digits 0–9)
    net = neural_network.Network([784, 30, 10])

    # Train for 30 epochs, batch size 10, learning rate 3.0
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == "__main__":
    main()


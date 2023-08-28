import numpy as np
from loss_functions import mean_squared_error
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

class Dense_layer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(1, output_size)

    def forward_prop(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output
    
    def backward_prop(self, error_to_output, learn_rate):
        error_to_input = np.dot(error_to_output, self.weights.T)
        error_to_weights = np.dot(self.input.T, error_to_output)
        error_to_biases = error_to_output

        self.weights -= learn_rate * error_to_weights
        self.biases -= learn_rate * error_to_biases
        return error_to_input

class Activation_layer(Layer):
    def __init__(self, activation_func, activation_func_derivative):
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative

    def forward_prop(self, input):
        self.input = input
        self.output = self.activation_func(input)
        return self.output
    
    def backward_prop(self, error_to_output, learn_rate):
        return self.activation_func_derivative(self.input) * error_to_output
    
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def add_layers(self, layers):
        self.layers = layers
    
    def add_layer(self,layer):
        self.layers.append(layer)

    def choose_loss_func(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    def fit(self, x_train, y_train, epochs, learn_rate, print = True):
        for i in range(epochs):
            error_to_display = 0
            for j in range(len(x_train)):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                error_to_display += mean_squared_error(y_train[j], output)

                error = self.loss_derivative(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, learn_rate)
            if print:
                print('%d/%d mean error: %f' % (i+1, epochs, error_to_display/len(x_train)))
        return error_to_display
    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output
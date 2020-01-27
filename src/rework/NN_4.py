import numpy as np
from Functions import *


class Dense:

    def __init__(self, nodes, activation_function=None, input_nodes_size=None):

        self.nodes = nodes

        if activation_function:
            self.f = activation_function.f
            self.prime = activation_function.prime
        else:
            self.f = lambda _: _
            self.prime = lambda _: _

        self.input_nodes_size = input_nodes_size
        self.weights = None
        self.biases = None

    def layer_output(self, activation):

        return self.f(np.dot(activation, self.weights) - self.biases)

    def set_weights(self, prev_layer_output=None):

        self.biases = np.random.rand(self.nodes)
        self.weights = np.random.rand(prev_layer_output, self.nodes) if prev_layer_output \
            else np.random.rand(self.input_nodes_size, self.nodes)


class FeedForward:

    def __init__(self, seed=None):

        self.seed = seed
        self.layers = []
        self.no_of_layers = 0

    def add(self, layer):

        if not len(self.layers) and not layer.input_nodes_size:
            raise Exception("The first layer requires an input shape")

        self.layers.append(layer)
        self.no_of_layers += 1

    def compile(self, error_function=None):

        self.layers[0].set_weights()
        for idx, layer in enumerate(self.layers[1::]):
            layer.set_weights(self.layers[idx].nodes)

        if not error_function:
            self.error_function = lambda target, actual: -(target - actual)
        self.MSE = lambda target, actual: np.sum(np.square(target-actual))

    def forward_propagation(self, x_in):

        hidden = [x_in]
        for layer in self.layers:
            hidden.append(layer.layer_output(hidden[-1]))

        return hidden

    def back_propagation(self, example, target, alpha=None):

        if alpha:
            self.alpha = alpha


        # New adjustments weights matrices
        weight_adjustment = [np.zeros(layer.weights.shape, dtype=np.float64)[0] for layer in self.layers]
        bias_adjustments = [np.zeros(layer.biases.shape, dtype=np.float64)[0] for layer in self.layers]

        # using each example, compute a gradient
        activations = self.forward_propagation(example)
        error = self.error_function(target,activations[-1])
        # compute delta for output layer
        deltas = [error * self.layers[-1].prime(activations[-1])]

        for layer, activation in zip(self.layers[::-1], activations[::-1]):
            # print(layer.weights.shape)
            print(activation.shape,activation.shape)
            deltas.append(deltas[-1].dot(layer.weights.T) * layer.prime(activation[np.newaxis,:]))
            # deltas.append( layer.prime(activation))
        print(deltas)
        deltas.reverse()


        # compute the adjustments for each weight
        for activation, layer_adjustment, delta in zip(activations, weight_adjustment, deltas):
            layer_adjustment += activation.T.dot(delta)

        self.adjust_weights(weight_adjustment)
        return self.MSE(target, self.predict(example))
        # err += self.error(target, activations[-1])

        # add the adjustments back on to the weights
        # for i in range(len(self.weights)):
        #     self.weights[i] -= self.alpha * (weight_adjustment[i ] /b_s)

    def adjust_weights(self, adjustment_layers):

        for layer, adjustment in zip(self.layers, adjustment_layers):
            layer.weights -= .1 * adjustment

    def predict(self, model_input):
        return self.forward_propagation(model_input)[-1]







# xor example

if __name__ == '__main__':

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    ff = FeedForward()
    ff.add(Dense(2, activation_function=Tanh, input_nodes_size=2))
    ff.add(Dense(1))
    ff.compile()
    # print(ff.forward_propagation(X[2]))
    for i in range(2):
        print(ff.back_propagation(X[2], Y[2]))
    for xor_input in X:
        print(ff.predict(xor_input))



    # nn = neural_net([2, 2, 1], seed=0)
    # nn.load("trained2")


from testing_data import labeled_data
from Processing import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

class neural_net:

    def __init__(self, layers=[18,9], alpha=1e-1, name="NN", seed=None):
        self.seed = seed
        self.name = name
        self.alpha = alpha

        self.f = lambda x: np.tanh(x)
        self.f_prime = lambda x: (1 + self.f(x)) * (1 - self.f(x))
        # self.f = lambda x: np.maximum(0,x)
        # self.f_prime = lambda x: (x>0)*1.0

        self.f_out = np.vectorize(lambda x: 1 if x > .5 else 0)
        self.fout_prime = np.vectorize(lambda x: (x>0)*1)

        self.s = lambda x: 1 / (1 + np.exp(-x))
        self.s_prime = lambda x: np.multiply(self.s(x), 1.0 - self.s(x))


        self.error = lambda x,y: np.square(x - y).sum()

        self.layers = layers
        self.nn_length = len(layers)
        self.weights = []
        self.c_error = []
        self.acc = []
        for layer in range(self.nn_length - 1):
            if seed:
                np.random.seed(seed)
            w = np.random.rand(layers[layer] + 1, layers[layer + 1])*2
            self.weights.append(w)

    # functions to save/load weights for later use
    def save(self, file=None):
        if not file:
            file=self.name
        np.save(file, np.array(self.weights))

    def load(self, file):
        self.weights = np.load(file + ".npy", allow_pickle=True)
        self.layers = [len(i)-1 for i in self.weights]
        self.layers.append(len(self.weights[-1][0]))
        self.nn_length = len(self.layers)
        print("Loading success")
        print(f"NN structure: {self.layers}")

    def forward_propagation(self, x_in):
        hidden = [x_in]
        for i in range(len(self.weights)-1):
            # Matrix multiplication in forward propagation
            x_in = np.dot(hidden[i], self.weights[i])
            activated = self.f(x_in)
            activated = np.concatenate((np.ones(1), np.array(activated, dtype=np.float)))
            hidden.append(activated)

        # Compute last layer output
        activated = self.f(np.dot(hidden[-1], self.weights[-1]))
        hidden.append(activated)
        return hidden

    def batch_back_propagation(self, examples, targets, alpha=None):
        if alpha:
            self.alpha = alpha
        # New adjustments weights matrices
        adjustment = [np.zeros(layer.shape, dtype=np.float64) for layer in self.weights]
        b_s = len(examples)
        err = 0

        # using each example, compute a gradient
        for example, target in zip(examples, targets):
            activations = self.forward_propagation(example)
            error = -(target-activations[-1])
            # compute delta for output layer
            delta = [error * self.f_prime(activations[-1])]

            # need to propagate errors from last layer forward
            for i in range(self.nn_length - 2, 0, -1):
                # ignore first index to account for biases
                delta.append(delta[-1].dot(self.weights[i][1:].T) * self.f_prime(activations[i][1:]))

            delta.reverse()

            # compute the adjustments for each weight
            for i in range(len(self.weights)):
                adjustment[i] += activations[i][np.newaxis,:].T.dot(delta[i][np.newaxis,:])

            err += self.error(target, activations[-1])

        # add the adjustments back on to the weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * (adjustment[i]/b_s)

        self.c_error.append(err/b_s)


    def batch_fit(self, data, targets, alpha=.5, batch_size=10, epochs=10000, X=None, Y=None):

        ones = np.ones((1, data.shape[0]))

        #add bias to input
        x_p = np.concatenate((ones.T, data), axis=1)
        batch = int(len(x_p)/batch_size)
        a, b = shuffle(x_p, targets)
        split_x = np.split(a, batch)
        split_y = np.split(b, batch)


        for k in range(epochs+1):
            # some remnant code I used to test batching the entire data set ensuring
            # each data was used to train
            # a, b = shuffle(x_p, targets)
            sample = np.random.randint(len(split_x))
            # self.batch_back_propagation(split_x[sample], split_y[sample], alpha=alpha)
            # split_x = np.split(a, batch)
            # split_y = np.split(b, batch)
            # for sample in range(len(split_x)):
            # print(f"Accuracy: {self.acc[-1]:.5f} %")
            # Batch back propagation
            # every 2000 epochs the accuracy is computed by comparing the
            # best guesses of the NN vs the actual computed XOR answers in the
            # labeled data set
            self.batch_back_propagation(split_x[sample], split_y[sample], alpha=alpha)
            if not k % 2000:
                print(f"Epoch: {k} Error: {self.c_error[-1]}")
            if not k % 2000 and X is not None:
                print(f"Computing accuracy on {len(X)} samples...")
                self.acc.append(self.accuracy(X, Y))
                print(f"Accuracy: {self.acc[-1]:.5f} %")



    def accuracy(self, X, Y):
        # Compute the accuracy by comparing the nn prediction
        # against the labelled data using the training data
        acc = 0
        for x, y in zip(X, Y):
            acc += process_move(y) == process_move(self.prediction(x))
        acc = acc/len(X)*100
        return acc

    def prediction(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0, len(self.weights)):
            val = self.f(np.dot(val, self.weights[i]))
            val = np.concatenate((np.ones(1).T, np.array(val)))
        # the output uses a function to force outputs into
        # 1s and 0s
        val = self.f_out(val)
        return val[1:]




# xor example


if __name__ == '__main__':
    # x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 1, 1, 0])
    # nn = neural_net([2, 2, 1], seed=0)
    nn = neural_net(layers=[18,60,60,9],name="t2", seed=None)
    # nn.load("trained2")

    x, y = labeled_data()
    x_, y_ = shuffle(x, y)
    train_len = 245000
    x, y = x_[0:train_len], y_[0:train_len]
    X = x_[train_len::]
    Y = y_[train_len::]

    i=True
    while(i != ("n" or "no")):
        print("Training...")
        nn.batch_fit(x, y, X=X,Y=Y, batch_size=10, alpha=.1, epochs=50000)
        i = input("Continue?(Enter or 'n')")
    #
    # for i in x:
    #     print(nn.prediction(i))

    # Demonstrate predictions for the first 50 examples in the
    # testing set
    for idx, s in enumerate(X[0:50]):
        o = nn.prediction(s)
        print(f"Piles: {bin_to_list(s)} NN prediction: {process_move(o)} "
              f"algorithm output: {process_move(Y[idx])}")
    print(f"Computing accuracy on {len(X)} samples...")
    print(f"Accuracy: {nn.accuracy(X, Y):.5f} %")

    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title(f"Error vs Epoch: Alpha={nn.alpha}")
    plt.xlabel("Training Iterations")
    plt.ylabel("Relative Error")
    plt.plot(nn.c_error)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.title(f"Accuracy: Alpha={nn.alpha}")
    plt.xlabel("Training Iterations*100")
    plt.ylabel("Accuracy")
    plt.plot(nn.acc)
    plt.grid(True)
    plt.tight_layout()
    plt.show()                  
import numpy as np


class Tanh:

    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def prime(x):
        return (1 + Tanh.f(x)) * (1 - Tanh.f(x))


class Relu:

    @staticmethod
    def f(x):
        return np.maximum(0, x)

    @staticmethod
    def prime(x):
        return (x > 0)*1.0
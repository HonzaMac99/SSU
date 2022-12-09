import numpy as np

class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):

        return  np.exp(X.T - np.max(X, axis=1)).T / np.sum(e_X, axis=1, keepdims=True)

    def delta(self, Y, delta_next):

        return Y*(delta_next - np.sum((delta_next * Y), axis=1)[:, np.newaxis])



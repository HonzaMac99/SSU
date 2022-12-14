import numpy as np

class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        # subtracting max values to prevent overflows
        X = np.exp(X-np.vstack(np.max(X, axis=1)))
        # X = np.exp(X)

        norm_X = np.sum(X, axis=1)
        return X/np.vstack(norm_X)

    def delta(self, Y, delta_next):
        Y_d = Y*(1-Y)
        return delta_next*Y_d


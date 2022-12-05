import numpy as np

class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        norm = 0
        for i in range(len(X)):
            norm += np.exp(X[i])
        for i in range(len(X)):
            X[i] = np.exp(X[i])/norm
        return X

    def delta(self, Y, delta_next):
        return delta_next*()
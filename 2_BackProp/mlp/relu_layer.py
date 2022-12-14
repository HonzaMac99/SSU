import numpy as np

class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        return np.maximum(0, X)

    def delta(self, Y, delta_next):
        # the derivative is undefined in Y=0
        #  -> choosing value 0 if Y=0
        Y_d = np.where(Y > 0, 1, 0)
        return delta_next*Y_d

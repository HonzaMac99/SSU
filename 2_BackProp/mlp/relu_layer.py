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
        # the derivative is undef in Y=0
        #  -> choosed 1 if Y=0
        delta_Y = np.array([(1 if Y[i] >= 0 else 0) for i in range(len(Y))])
        return delta_next.T@delta_Y

import numpy as np
class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):

        return rnp.maximum(0, X)

    def delta(self, Y, delta_next):
        return delta_next * (np.where(Y>0,1,0))


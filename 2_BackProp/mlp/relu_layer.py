class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        return max(0, X)

    def delta(self, Y, delta_next):
        # the derivative is undef in Y=0
        #  -> choosed 1 if Y=0
        return delta_next*(1 if Y >= 0 else 0)

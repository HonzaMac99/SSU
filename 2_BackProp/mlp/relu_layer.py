class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        pass  # TODO IMPLEMENT

    def delta(self, Y, delta_next):
        pass  # TODO IMPLEMENT

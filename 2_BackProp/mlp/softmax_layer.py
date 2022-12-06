class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        norm = np.sum(np.exp(X))
        X = np.exp(X)/norm
        return X

    def delta(self, Y, delta_next):
        norm = np.sum(np.exp(X))
        Y = (Y*norm - Y)/(norm**2)
        return delta_next.T@Y
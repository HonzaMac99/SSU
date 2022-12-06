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
        norm = 0
        for i in range(len(Y)):
            norm += np.exp(Y[i])
        for i in range(len(Y)):
            Y[i] = (Y[i]*norm - Y[i])/(norm**2)
        return delta_next.T@Y
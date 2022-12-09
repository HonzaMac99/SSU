import numpy as np

class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        n_inputs = X.shape[1]
        loss = -np.sum(T*np.log(X) + (1-T)*np.log(1-X), axis=1)/n_inputs
        # loss = -np.sum(T * np.log(X), axis=1)/n_inputs

        # print("Loss.forward")
        # print(X)
        # print(loss)
        # print("")

        return loss

    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        loss_delta = -(T-X)/(X*(1-X))
        # loss_delta = -T/X

        # gradient clipping
        treshold = 100
        loss_delta = np.clip(loss_delta, -treshold, treshold)

        # print("Loss.delta")
        # print(T)
        # print(X)
        # print(loss_delta)
        # print("")

        return loss_delta


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name

    def forward(self, X, T):

        # softmax
        n_inputs = X.shape[1]
        X = np.exp(X-np.vstack(np.max(X, axis=1)))
        norm_X = np.sum(X, axis=1)
        X /= np.vstack(norm_X)

        # loss = -np.sum(T*np.log(X) + (1-T)*np.log(1-X), axis=1)/n_inputs
        loss = -np.sum(T * np.log(X), axis=1)/n_inputs

        # print("Loss.forward")
        # print(X)
        # print(loss)
        # print("")

        return loss

    def delta(self, X, T):

        # loss_delta = -(T-X)/(X*(1-X))
        loss_delta = -T/X
        # X_d = X*(1-X)
        # soft_loss_delta = loss_delta*X_d
        # treshold = 100
        # soft_loss_delta = np.clip(soft_loss_delta, -treshold, treshold)

        # print("SoftLoss.delta")
        # print(T)
        # print(X)
        # print(soft_loss_delta)
        # print("")

        # return soft_loss_delta

        delta = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
        diag = np.arange(X.shape[1])
        delta[:, diag, diag] = X
        delta -= np.einsum('bi, bo->bio', X, X, optimize="True")

        return np.einsum('brc, br->bc', delta, loss_delta, optimize="True")


import numpy as np
import matplotlib.pyplot as plt

class SoftmaxClassifier(object):
    def __init__(self, nclasses, epochs=300, stepsize=1, regularisation=0.001):
        """Initialise a Softmax Classifier model

        Parameters:
            - nclasses : Number of unique classes to predict
            - epochs   : number of training iterations to make
                         (default = 300)
            - stepsize : for gradient descent, how large to make each
                         step (default = 1)
            - regularisation : lambda factor in loss calculation.
                               Larger values punish large weights
                               more, can lead to less learning.
                               (default = 0.001)
        """
        print("Making a Softmax linear classifier")
        self.nclasses = nclasses
        self.epochs = epochs  # number of training iterations
        self.stepsize = stepsize  # gradient optimisation step size
        self.reg = regularisation  # regularisation strength
        self._training_loss = []
        return

    def _setup_parameters(self, ndims):
        self.W = 0.01 * np.random.randn(ndims, self.nclasses)  # weights
        self.b = np.zeros((1 ,self.nclasses))  # biases
        return

    def _find_loss(self, probs, n_entries, y):
        """
        Use cross-entropy loss (i.e. softmax) plus regularisation
        L = 1/N * \sum_i{L_i} + 1/2 * \lambda * \sum_k{\sum_l{W^2_{k,l}}}
        """
        logprobs = -np.log(probs[range(n_entries), y])
        data_loss = np.sum(logprobs) / n_entries
        reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
        return data_loss + reg_loss

    def _back_propagate(self, probs, x, y):
        """Do backpropagation.
        Gradient for scores, for class k, is just
          > dL_k/df_k = p_k - 1[y_i == k]
        normalised by N) - i.e. is only probability-1 for correct class

        Parameters:
            - probs : normalised class probabilites
            - x : training data
            - y : labels for x

        Returns:
            - dW : gradient for weights
            - db : gradient for bias
        """
        n_entries = x.shape[0]

        dscores = probs
        dscores[range(n_entries), y] -= 1
        dscores /= n_entries

        # backpropagate gradient into weights/bias
        # ds/dW = x
        # ds/db = 1
        dW = np.dot(x.T, dscores)
        dW += self.reg * self.W  # don't forget the regularisation!
        db = np.sum(dscores, axis=0, keepdims=True)
        return dW, db

    def fit(self, x, y):
        """Train classifier on a dataset

        Parameters:
            - x : unlabelled training data
            - y : labels for x
        """
        self._setup_parameters(x.shape[1])
        n_entries = x.shape[0]
        for i in range(self.epochs):

            # calculate scores (s = Wx + b)
            scores = np.dot(x, self.W) + self.b

            # calculate class probabilities (normalised)
            #  p_k = e^{f_k} / \sum_j{e^f_j}
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # find loss for this iteration
            loss = self._find_loss(probs, n_entries, y)
            self._training_loss.append(loss)
            if i % 10 == 0:
                print("Epoch {} loss: {:.03f}".format(i, loss))

            # back propagate gradients
            dW, db = self._back_propagate(probs, x, y)

            # then update the parameters
            self.W += -self.stepsize * dW
            self.b += -self.stepsize * db

        print("Training accuracy: {:.02f}".format(self.accuracy(x, y)))
        return

    def predict(self, x):
        """
        Predict classes for dataset x

        Parameters:
            - x : dataset to predict

        Returns:
            - predicted : array containing predicted classes for x
        """
        scores = np.dot(x, self.W) + self.b
        predicted = np.argmax(scores, axis=1)
        return predicted

    def accuracy(self, x, y):
        """Find accuracy of classifier on dataset x

        Parameters:
            - x : dataset to find accuracy for
            - y : class labels for x

        Returns:
            - accuracy
        """
        predicted = self.predict(x)
        return np.mean(predicted == y)

    def plot_boundaries(self, x, y, d1=0, d2=1):
        """
        Plot 2D decision boundaries for dataset x

        Parameters:
            - x : dataset to use to plot decision boundaries
            - y : class labels for x
            - d1 : Index for first dimension (optional, default=0)
            - d2 : Index for second dimension (optional, default=1)

        Returns:
            - fig : matplotlib figure
        """
        h = 0.02
        x_min, x_max = x[:, d1].min() - 1, x[:, d1].max() + 1
        y_min, y_max = x[:, d2].min() - 1, x[:, d2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        z = np.dot(np.c_[xx.ravel(), yy.ravel()], self.W) + self.b
        z = np.argmax(z, axis=1)
        z = z.reshape(xx.shape)
        fig = plt.figure()
        plt.contourf(xx, yy, z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(x[:, d1], x[:, d2], c=y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        return fig

    def plot_training_loss(self):
        """Plot loss during training as function of epoch"""
        fig = plt.figure()
        plt.plot(range(self.epochs), self._training_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Training loss')
        return fig
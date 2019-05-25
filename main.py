import numpy as np
from data import make_spiral_data, plot_2d_data
from softmax import SoftmaxClassifier


if __name__ == '__main__':

    # generate spiral data
    n_classes = 3
    data, y = make_spiral_data(100, n_classes, 2)
    fig = plot_2d_data(data, y)
    fig.show()

    # train model
    model = SoftmaxClassifier(n_classes)
    model.fit(data, y)
    print("Training accuracy is {:0.2f}".format(model.training_accuracy))
    print("Training loss is {:0.3f}".format(model.training_loss))
    boundaries = model.plot_boundaries(data, y)
    boundaries.show()
    loss_vs_epoch = model.plot_training_loss()
    loss_vs_epoch.show()

    # test model against 'new' data
    new_data, new_y = make_spiral_data(50, n_classes, 2)
    predictions = model.predict(new_data)
    print("Test accuracy is {:0.2f}".format(model.accuracy(new_data, new_y)))


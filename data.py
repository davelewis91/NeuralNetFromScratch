import numpy as np
import matplotlib.pyplot as plt


def make_spiral_data(N, classes = 3, dimensions = 2):
    """Generates new dataset based on the 'spiral' distribution.

    Parameters:
        - N          : number of datapoints (per class)
        - classes    : number of classes
        - dimensions : number of features per datapoint

    Returns:
         - x : unlabelled data
         - y : labels for x
    """
    K = classes
    D = dimensions
    x = np.zeros((N*K, D))
    y = np.zeros(N*K, dtype='uint8')

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0, 1, N)  # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # theta
        x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return x, y

def plot_2d_data(x, y=None, path=None, labels=None):
    """Plots 2D data, optionally colouring by class.

    Parameters:
        - x : unlabelled data to plot
        - y : labels for data (default None)
        - path: filepath to save figure (optional)

    Returns:
        - fig : matplotlib figure
    """
    fig = plt.figure()
    if any(y): cmap = plt.cm.Spectral
    else: cmap = None
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=cmap)
    if not labels:
        plt.xlabel('dimension 1')
        plt.ylabel('dimension 2')
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    if path:
        plt.savefig(path, bbox_inches='tight')
    return fig
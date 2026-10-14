import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        xtr = np.asarray(xtr)
        ytr = np.asarray(ytr).ravel() 

        self._class_labels = np.unique(ytr)
        centroids_list = []
        for c in self._class_labels:
            centroid_c = xtr[ytr == c].mean(axis=0)
            centroids_list.append(centroid_c)
        self._centroids = np.array(centroids_list)
        return self

    def predict(self, xts):
        pass

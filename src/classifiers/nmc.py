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
        pass

    def predict(self, xts):
        xts = np.asarray(xts)
        n_samples = xts.shape[0]

        # Se il modello è stato addestrato, usa le etichette reali
        if self._class_labels is not None and self._centroids is not None:
            distances = euclidean_distances(xts, self._centroids)
            closest_centroid_indices = np.argmin(distances, axis=1)
            yc = self._class_labels[closest_centroid_indices]
        else:
            # Se il modello NON è addestrato, restituisce una classe predefinita (es. 0) per ogni campionamento
            yc = np.zeros(n_samples, dtype=int)

        # ravel si assicura che l'array sia unidimensionale (ndim == 1) e di lunghezza x.shape[0]
        return yc.ravel()

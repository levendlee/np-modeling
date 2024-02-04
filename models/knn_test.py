# KNN test.

import unittest

import numpy as np

import sklearn.neighbors
from models.knn import Algorithm, KNearestNeighbours

np.random.seed(202402)


class KNearestNeighboursTest(unittest.TestCase):
    def test_predict(self):
        n_train_samples = 100
        n_test_samples = 10
        n_features = 128
        k = 5
        n_classes = 3

        x_train = np.random.normal(size=[n_train_samples, n_features]).astype(
            np.float32)
        y_train = np.random.randint(0,
                                    n_classes,
                                    size=[n_train_samples],
                                    dtype=np.int32)
        x_test = np.random.normal(size=[n_test_samples, n_features]).astype(
            np.float32)

        knn = KNearestNeighbours(x_train=x_train,
                                 y_train=y_train,
                                 k=k,
                                 n_classes=n_classes,
                                 algorithm=Algorithm.Distance)
        y_test = knn.predict(x_test)
        print('Distance: ', y_test)

        neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,
                                                       weights='distance')
        neigh.fit(x_train, y_train)
        sklearn_y_test = neigh.predict(x_test)
        print('Sklearn distance: ', sklearn_y_test)
        # np.testing.assert_array_equal(y_test, sklearn_y_test)

        knn = KNearestNeighbours(x_train=x_train,
                                 y_train=y_train,
                                 k=k,
                                 n_classes=n_classes,
                                 algorithm=Algorithm.Uniform)
        y_test = knn.predict(x_test)
        print('Uniform: ', y_test)

        neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,
                                                       weights='uniform')
        neigh.fit(x_train, y_train)
        sklearn_y_test = neigh.predict(x_test)
        print('Sklearn uniform: ', sklearn_y_test)

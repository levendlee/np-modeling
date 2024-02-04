"""
KNN.
"""

import dataclasses
import enum
import numpy as np


def elucidian_distance(x_test: np.ndarray,
                       x_train: np.ndarray,
                       broadcast: bool = False) -> np.ndarray:
    # x_test:  [M, D]
    # x_train: [N, D]
    # diff2:   [M, N, D]
    # output:  [M, N]
    if broadcast:
        diff2 = (np.expand_dims(x_test, axis=1) -
                 np.expand_dims(x_train, axis=0))**2
        return np.sqrt(np.sum(diff2, axis=2))
    else:
        # sum(||x_train - x_test||^2)
        # sum(x_train^2)  - 2 * x_train @ x_test.T + sum(x_test^2)
        return np.sqrt(
            np.sum(x_test**2, axis=1, keepdims=True) + 2 * x_test @ x_train.T +
            np.expand_dims(np.sum(x_train**2, axis=1), axis=0))


class Algorithm(enum.Enum):
    Uniform = 0
    Distance = 1


@dataclasses.dataclass
class KNearestNeighbours:

    x_train: np.ndarray
    y_train: np.ndarray
    k: int
    n_classes: int
    algorithm: Algorithm

    def kneighbours(self, x_test: np.ndarray) -> np.ndarray:
        distance = elucidian_distance(x_test, self.x_train)
        print('distance: ', distance.shape)
        topk_indices = np.argpartition(distance, self.k, axis=-1)[:, :self.k]
        print('topk_indices: ', topk_indices.shape)
        topk_distances = np.take_along_axis(distance, topk_indices, axis=1)
        print('topk_distances: ', topk_indices.shape)
        # output: [M, K]
        return topk_indices, topk_distances

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self.algorithm == Algorithm.Uniform:
            batched_topk_indices, _ = self.kneighbours(x_test)
            return np.apply_along_axis(lambda indices: np.argmax(
                np.bincount(self.y_train[indices], minlength=self.n_classes)),
                                       axis=1,
                                       arr=batched_topk_indices)
        else:
            assert self.algorithm == Algorithm.Distance
            batched_topk_indices, batched_topk_distances = self.kneighbours(
                x_test)
            batched_topk_inv_distances = 1.0 / batched_topk_distances
            batched_topk_inv_distance_sum = np.sum(batched_topk_inv_distances,
                                                   axis=1,
                                                   keepdims=True)
            batched_topk_weights = (batched_topk_inv_distances /
                                    batched_topk_inv_distance_sum)
            # [M, K] -> [M, K, 1] -> [M, K, C]
            batch_size = batched_topk_indices.shape[0]
            class_weights = np.zeros([batch_size * self.k, self.n_classes])
            class_weights[np.arange(batch_size * self.k),
                          self.y_train[batched_topk_indices.
                                       flat]] = batched_topk_weights.flat
            class_weights = class_weights.reshape(
                [batch_size, self.k, self.n_classes])
            return np.argmax(np.sum(class_weights, axis=1), axis=1)

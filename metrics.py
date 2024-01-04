# Evaluation metrics

import dataclasses

import numpy as np

@dataclasses.dataclass
class BinaryClassificationMetrics:
    precision: float
    recall: float

def calculate_binary_classification_metrics(predicts, labels):
    tp = np.sum(predicts and predicts == labels)
    fp = np.sum(predicts and predicts != labels)
    fn = np.sum(labels and predicts != labels)
    return BinaryClassificationMetrics(precision=(tp/(tp + fp)),
                                       recall=(tp/(tp+fn)))
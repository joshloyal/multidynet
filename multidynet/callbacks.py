from .multidynet import calculate_probabilities
from .metrics import calculate_auc, calculate_correlation


class TestMetricsCallback(object):
    def __init__(self, Y, probas, test_indices):
        self.test_indices_ = test_indices
        self.Y_ = Y
        self.probas_ = probas
        self.aucs_ = []
        self.correlations_ = []

    def __call__(self, model, Y):
        probas = calculate_probabilities(
                model.X_, model.lambda_, model.delta_)
        self.aucs_.append(calculate_auc(self.Y_, probas, self.test_indices_))
        self.correlations_.append(calculate_correlation(
            self.probas_, probas, self.test_indices_))

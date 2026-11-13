"""Select one post-update partition per seed using reference-label ACC."""

import numpy as np

from .data import count_cluster_matches, evaluate_clustering


class PartitionSelection:

    def __init__(self, reference_labels):
        self.reference_labels = np.array(reference_labels, copy=True)
        self.matched_count = -1
        self.selected_labels = None

    def observe(self, predicted_labels):
        count = count_cluster_matches(self.reference_labels, predicted_labels)
        if count > self.matched_count:
            self.matched_count = count
            self.selected_labels = np.array(predicted_labels, copy=True)

    def scores(self):
        if self.selected_labels is None:
            raise RuntimeError("Training produced no partition to evaluate")
        metrics = evaluate_clustering(self.reference_labels, self.selected_labels)
        return {key: metrics[key] for key in ("ACC", "NMI", "ARI")}

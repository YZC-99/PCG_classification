"""
Sensitive,Specificity,Precision,Accuracy,F1-score
"""
from torchmetrics import Recall,Specificity,Precision,Accuracy,F1Score

class Evaluate_get_metrics(object):
    def __init__(self):
        self.accuracy = Accuracy()
        self.precision = Precision(average='macro', num_classes=2)
        self.specificity = Specificity(average='macro', num_classes=2)
        self.f1 = F1Score(average='macro', num_classes=2)
        self.recall = Recall(average='macro', num_classes=2)  # Recall is the same as sensitivity

    def add(self, preds, labels):
        """
        Add new predictions and labels for evaluation.

        :param preds: Predicted labels (batch_size,)
        :param labels: True labels (batch_size,)
        """
        # Ensure preds and labels are torch tensors
        preds, labels = preds, labels
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.specificity.update(preds, labels)
        self.f1.update(preds, labels)
        self.recall.update(preds, labels)

    def reset(self):
        """Reset the states of all metrics."""
        self.accuracy.reset()
        self.precision.reset()
        self.specificity.reset()
        self.f1.reset()
        self.recall.reset()

    def compute(self):
        """Compute and return the evaluation metrics."""
        metrics = {
            'Sensitivity': self.recall.compute().item(),
            'Specificity': self.specificity.compute().item(),
            'Precision': self.precision.compute().item(),
            'Accuracy': self.accuracy.compute().item(),
            'F1-score': self.f1.compute().item()
        }
        return metrics
"""Results saving callback."""
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import json
import time
import torch
from sklearn.metrics import classification_report


class ResultsSaver(Callback):
    """
    Callback to save test results in a single comprehensive JSON file.

    Saves only test_results.json with:
        - timestamp
        - test metrics (loss, accuracy, precision, recall, f1)
        - classification_report (formatted string)
        - confusion_matrix
        - per_class_metrics (precision, recall, f1, support per class)

    Args:
        save_dir: Directory to save results
    """

    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.test_preds = []
        self.test_targets = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect predictions and targets during test."""
        x, y = batch
        logits = pl_module(x)
        preds = torch.argmax(logits, dim=1)

        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())

    def on_test_end(self, trainer, pl_module):
        """Generate comprehensive test results JSON."""
        metrics = trainer.callback_metrics

        # Concatenate all predictions and targets
        all_preds = torch.cat(self.test_preds).numpy()
        all_targets = torch.cat(self.test_targets).numpy()

        # Compute confusion matrix from predictions
        from sklearn.metrics import confusion_matrix
        num_classes = pl_module.hparams.output_dim
        labels = list(range(num_classes))
        cm = confusion_matrix(all_targets, all_preds, labels=labels)
        confusion_matrix_list = cm.tolist()

        # Generate classification report
        # Use labels parameter to handle cases where some classes may not appear in test data
        class_names = [str(i) for i in range(num_classes)]
        clf_report_str = classification_report(all_targets, all_preds, labels=labels, target_names=class_names, zero_division=0)
        clf_report_dict = classification_report(all_targets, all_preds, labels=labels, target_names=class_names, output_dict=True, zero_division=0)

        # Extract per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if class_name in clf_report_dict:
                per_class_metrics[class_name] = {
                    "precision": clf_report_dict[class_name]["precision"],
                    "recall": clf_report_dict[class_name]["recall"],
                    "f1-score": clf_report_dict[class_name]["f1-score"],
                    "support": clf_report_dict[class_name]["support"]
                }

        # Build comprehensive results
        test_results = {
            "timestamp": time.time(),
            "test_loss": float(metrics.get('test/loss', 0)),
            "test_accuracy": float(metrics.get('test/accuracy', 0)) * 100,  # Convert to percentage
            "precision": float(metrics.get('test/precision', 0)),
            "recall": float(metrics.get('test/recall', 0)),
            "f1_score": float(metrics.get('test/f1', 0)),
            "classification_report": clf_report_str,
            "confusion_matrix": confusion_matrix_list,
            "per_class_metrics": per_class_metrics
        }

        # Save to single JSON file
        results_file = self.save_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        # Clear stored data
        self.test_preds = []
        self.test_targets = []

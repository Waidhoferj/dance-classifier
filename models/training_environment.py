import importlib
from models.utils import calculate_metrics, plot_to_image, get_dance_mapping
import numpy as np
from abc import ABC, abstractmethod
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class TrainingEnvironment(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: dict,
        learning_rate=1e-4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = config["training_environment"].get(
            "learning_rate", learning_rate
        )
        self.experiment_loggers = load_loggers(
            config["training_environment"].get("loggers", {})
        )
        self.config = config
        self.has_multi_label_predictions = not (
            type(criterion).__name__ == "CrossEntropyLoss"
        )
        self.save_hyperparameters(
            {
                "model": type(model).__name__,
                "loss": type(criterion).__name__,
                "config": config,
                **kwargs,
            }
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.TensorType], batch_index: int
    ) -> torch.Tensor:
        features, labels = batch
        outputs = self.model(features)
        if self.has_multi_label_predictions:
            outputs = nn.functional.sigmoid(outputs)
        loss = self.criterion(outputs, labels)
        metrics = calculate_metrics(
            outputs,
            labels,
            prefix="train/",
            multi_label=self.has_multi_label_predictions,
        )
        self.log_dict(metrics, prog_bar=True)
        experiment = self.logger.experiment
        for logger in self.experiment_loggers:
            logger.step(experiment, batch_index, features, labels)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.TensorType], batch_index: int
    ):
        x, y = batch
        preds = self.model(x)
        if self.has_multi_label_predictions:
            preds = nn.functional.sigmoid(preds)
        metrics = calculate_metrics(
            preds, y, prefix="val/", multi_label=self.has_multi_label_predictions
        )
        metrics["val/loss"] = self.criterion(preds, y)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.TensorType], batch_index: int):
        x, y = batch
        preds = self.model(x)
        if self.has_multi_label_predictions:
            preds = nn.functional.sigmoid(preds)
        metrics = calculate_metrics(
            preds, y, prefix="test/", multi_label=self.has_multi_label_predictions
        )
        if not self.has_multi_label_predictions:
            preds = nn.functional.softmax(preds, dim=1)
        y = y.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        # ROC-auc score
        try:
            metrics["test/roc_auc_score"] = torch.tensor(
                roc_auc_score(y, preds), dtype=torch.float32
            )
        except ValueError:
            # If there is only one class, roc_auc_score will throw an error
            pass

            pass
        self.log_dict(metrics, prog_bar=True)
        # Create confusion matrix

        preds = preds.argmax(axis=1)
        y = y.argmax(axis=1)
        cm = confusion_matrix(
            preds, y, normalize="all", labels=np.arange(len(self.config["dance_ids"]))
        )
        if hasattr(self, "test_cm"):
            self.test_cm += cm
        else:
            self.test_cm = cm

    def on_test_end(self):
        dance_ids = sorted(self.config["dance_ids"])
        np.fill_diagonal(self.test_cm, 0)
        cm = self.test_cm / self.test_cm.max()
        ConfusionMatrixDisplay(cm, display_labels=dance_ids).plot()
        image = plot_to_image(plt.gcf())
        image = torch.tensor(image, dtype=torch.uint8)
        image = image.permute(2, 0, 1)
        self.logger.experiment.add_image("test/confusion_matrix", image, 0)
        delattr(self, "test_cm")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }


class ExperimentLogger(ABC):
    @abstractmethod
    def step(self, experiment, data):
        pass


class SpectrogramLogger(ExperimentLogger):
    def __init__(self, frequency=100) -> None:
        self.frequency = frequency
        self.counter = 0

    def step(self, experiment, batch_index, x, label):
        if self.counter == self.frequency:
            self.counter = 0
            img_index = torch.randint(0, len(x), (1,)).item()
            img = x[img_index][0]
            img = (img - img.min()) / (img.max() - img.min())
            experiment.add_image(
                f"batch: {batch_index}, element: {img_index}", img, 0, dataformats="HW"
            )
        self.counter += 1


def load_loggers(logger_config: dict) -> list[ExperimentLogger]:
    loggers = []
    for logger_path, kwargs in logger_config.items():
        module_name, class_name = logger_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        Logger = getattr(module, class_name)
        loggers.append(Logger(**kwargs))
    return loggers

from models.utils import calculate_metrics


import pytorch_lightning as pl
import torch
import torch.nn as nn


class TrainingEnvironment(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: dict,
        learning_rate=1e-4,
        log_spectrograms=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.log_spectrograms = log_spectrograms
        self.config = config
        self.has_multi_label_predictions = (
            not type(criterion).__name__ == "CrossEntropyLoss"
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
        loss = self.criterion(outputs, labels)
        metrics = calculate_metrics(
            outputs,
            labels,
            prefix="train/",
            multi_label=self.has_multi_label_predictions,
        )
        self.log_dict(metrics, prog_bar=True)
        # Log spectrograms
        if self.log_spectrograms and batch_index % 100 == 0:
            tensorboard = self.logger.experiment
            img_index = torch.randint(0, len(features), (1,)).item()
            img = features[img_index][0]
            img = (img - img.min()) / (img.max() - img.min())
            tensorboard.add_image(
                f"batch: {batch_index}, element: {img_index}", img, 0, dataformats="HW"
            )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.TensorType], batch_index: int
    ):
        x, y = batch
        preds = self.model(x)
        metrics = calculate_metrics(
            preds, y, prefix="val/", multi_label=self.has_multi_label_predictions
        )
        metrics["val/loss"] = self.criterion(preds, y)
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.TensorType], batch_index: int):
        x, y = batch
        preds = self.model(x)
        self.log_dict(
            calculate_metrics(
                preds, y, prefix="test/", multi_label=self.has_multi_label_predictions
            ),
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

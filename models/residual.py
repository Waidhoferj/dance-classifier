import pytorch_lightning as pl
from pytorch_lightning import callbacks as cb
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchaudio
import yaml
from models.training_environment import TrainingEnvironment
from models.utils import LabelWeightedBCELoss
from preprocessing.dataset import DanceDataModule, get_datasets
from preprocessing.pipelines import (
    SpectrogramTrainingPipeline,
    WaveformPreprocessing,
)

# Architecture based on: https://github.com/minzwon/sota-music-tagging-models/blob/36aa13b7205ff156cf4dcab60fd69957da453151/training/model.py


class ResidualDancer(nn.Module):
    def __init__(self, n_channels=128, n_classes=50):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.res_layers = nn.Sequential(
            ResBlock(1, n_channels, stride=2),
            ResBlock(n_channels, n_channels, stride=2),
            ResBlock(n_channels, n_channels * 2, stride=2),
            ResBlock(n_channels * 2, n_channels * 2, stride=2),
            ResBlock(n_channels * 2, n_channels * 2, stride=2),
            ResBlock(n_channels * 2, n_channels * 2, stride=2),
            ResBlock(n_channels * 2, n_channels * 4, stride=2),
        )

        # Dense
        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.spec_bn(x)

        # CNN
        x = self.res_layers(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        # x = nn.Sigmoid()(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super().__init__()
        # convolution
        self.conv_1 = nn.Conv2d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(
            output_channels, output_channels, shape, padding=shape // 2
        )
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(
                input_channels,
                output_channels,
                shape,
                stride=stride,
                padding=shape // 2,
            )
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


def train_residual_dancer(config: dict):
    TARGET_CLASSES = config["dance_ids"]
    DEVICE = config["device"]
    SEED = config["seed"]
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(SEED, workers=True)
    feature_extractor = SpectrogramTrainingPipeline(**config["feature_extractor"])
    dataset = get_datasets(config["datasets"], feature_extractor)

    data = DanceDataModule(dataset, **config["data_module"])
    model = ResidualDancer(n_classes=len(TARGET_CLASSES), **config["model"])
    label_weights = data.get_label_weights().to(DEVICE)
    criterion = LabelWeightedBCELoss(label_weights)

    train_env = TrainingEnvironment(model, criterion, config)
    callbacks = [
        cb.EarlyStopping("val/loss", patience=2),
        cb.StochasticWeightAveraging(1e-2),
        cb.RichProgressBar(),
    ]
    trainer = pl.Trainer(callbacks=callbacks, **config["trainer"])
    trainer.fit(train_env, datamodule=data)
    trainer.test(
        train_env,
        datamodule=data,
    )

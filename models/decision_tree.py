import pytorch_lightning as pl
from sklearn.base import ClassifierMixin, BaseEstimator
import pandas as pd
from torch import nn
import torch
from typing import Iterator
import numpy as np
import json
from torch.utils.data import random_split
from tqdm import tqdm
import librosa
from joblib import dump, load
from os import path
import os

from preprocessing.dataset import get_music4dance_examples

DANCE_INFO_FILE = "data/dance_info.csv"
dance_info_df = pd.read_csv(
    DANCE_INFO_FILE,
    converters={"tempoRange": lambda s: json.loads(s.replace("'", '"'))},
)


class DanceTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Trains a series of binary classifiers to classify each dance when a song falls into its bpm range.

    Features:
        - Spectrogram
        - BPM
    """

    def __init__(self, device="cpu", lr=1e-4, verbose=True) -> None:
        self.device = device
        self.verbose = verbose
        self.lr = lr
        self.classifiers = {}
        self.optimizers = {}
        self.criterion = nn.BCELoss()

    def get_valid_dances_from_bpm(self, bpm: float) -> list[str]:
        mask = dance_info_df["tempoRange"].apply(
            lambda interval: interval["min"] <= bpm <= interval["max"]
        )
        return list(dance_info_df["id"][mask])

    def fit(self, x, y):
        """
        x: (specs, bpms). The first element is the spectrogram, second element is the bpm. spec shape should be (channel, freq_bins, sr * time)
        y: (batch_size, n_classes)
        """
        epoch_loss = 0
        pred_count = 0
        data_loader = zip(x, y)
        if self.verbose:
            data_loader = tqdm(data_loader, total=len(y))
        for (spec, bpm), label in data_loader:
            # find all models that are in the bpm range
            matching_dances = self.get_valid_dances_from_bpm(bpm)
            spec = torch.from_numpy(spec).to(self.device)
            for dance in matching_dances:
                if dance not in self.classifiers or dance not in self.optimizers:
                    classifier = DanceCNN().to(self.device)
                    self.classifiers[dance] = classifier
                    self.optimizers[dance] = torch.optim.Adam(
                        classifier.parameters(), lr=self.lr
                    )
            models = [
                (dance, model, self.optimizers[dance])
                for dance, model in self.classifiers.items()
                if dance in matching_dances
            ]
            for model_i, (dance, model, opt) in enumerate(models, start=1):
                opt.zero_grad()
                output = model(spec)
                target = torch.tensor([float(dance == label)], device=self.device)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()
                pred_count += 1
                loss.backward()
                if self.verbose:
                    data_loader.set_description(
                        f"model: {model_i}/{len(models)}, loss: {loss.item()}"
                    )
                opt.step()

    def predict(self, x) -> list[str]:
        results = []
        for spec, bpm in zip(*x):
            matching_dances = self.get_valid_dances_from_bpm(bpm)
            dance_i = torch.tensor(
                [self.classifiers[dance](spec) for dance in matching_dances]
            ).argmax()
            results.append(matching_dances[dance_i])
        return results

    def save(self, folder: str):
        # Create a folder
        classifier_path = path.join(folder, "classifier")
        os.makedirs(classifier_path, exist_ok=True)

        # Swap out model reference
        classifiers = self.classifiers
        optimizers = self.optimizers
        criterion = self.criterion

        self.classifiers = None
        self.optimizers = None
        self.criterion = None

        # Save the Pth models
        for dance, classifier in classifiers.items():
            torch.save(
                classifier.state_dict(), path.join(classifier_path, dance + ".pth")
            )

        # Save the Sklearn model
        dump(path.join(folder, "sklearn.joblib"))

        # Reload values
        self.classifiers = classifiers
        self.optimizers = optimizers
        self.criterion = criterion

    @staticmethod
    def from_config(folder: str, device="cpu") -> "DanceTreeClassifier":
        # load in weights
        model_paths = (
            p for p in os.listdir(path.join(folder, "classifier")) if p.endswith("pth")
        )
        classifiers = {}
        for model_path in model_paths:
            dance = model_path.split(".")[0]
            model = DanceCNN().to(device)
            model.load_state_dict(
                torch.load(path.join(folder, "classifier", model_path))
            )
            classifiers[dance] = model
        wrapper = load(path.join(folder, "sklearn.joblib"))
        wrapper.classifiers = classifiers
        return wrapper


class DanceCNN(nn.Module):
    def __init__(self, sr=16000, freq_bins=20, duration=6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        kernel_size = (3, 9)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 10)),
            nn.Conv2d(16, 32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 10)),
            nn.Conv2d(32, 32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 10)),
            nn.Conv2d(32, 16, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 10)),
        )

        embedding_dimension = 16 * 6 * 8
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten() if len(x.shape) == 3 else x.flatten(1)
        return self.classifier(x)


def features_from_path(
    paths: list[str], audio_window_duration=6, audio_duration=30, resample_freq=16000
) -> Iterator[tuple[np.array, float]]:
    """
    Loads audio and bpm from an audio path.
    """

    for path in paths:
        waveform, sr = librosa.load(path, mono=True, sr=resample_freq)
        num_frames = audio_window_duration * sr
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
        spec = librosa.feature.melspectrogram(y=waveform, sr=sr)
        spec_normalized = (spec - spec.mean()) / spec.std()
        spec_padded = librosa.util.fix_length(
            spec_normalized, size=sr * audio_duration, axis=1
        )
        batched_spec = np.expand_dims(spec_padded, axis=0)
        for i in range(audio_duration // audio_window_duration):
            spec_window = batched_spec[:, :, i * num_frames : (i + 1) * num_frames]
            yield (spec_window, tempo)


def train_decision_tree(config: dict):
    TARGET_CLASSES = config["global"]["dance_ids"]
    DEVICE = config["global"]["device"]
    SEED = config["global"]["seed"]
    SEED = config["global"]["seed"]
    EPOCHS = config["trainer"]["min_epochs"]
    song_data_path = config["data_module"]["song_data_path"]
    song_audio_path = config["data_module"]["song_audio_path"]
    pl.seed_everything(SEED, workers=True)

    df = pd.read_csv(song_data_path)
    x, y = get_music4dance_examples(
        df, song_audio_path, class_list=TARGET_CLASSES, multi_label=True
    )
    # Convert y back to string classes
    y = np.array(TARGET_CLASSES)[y.argmax(-1)]
    train_i, test_i = random_split(
        np.arange(len(x)), [0.1, 0.9]
    )  # Temporary to test efficacy
    train_paths, train_y = x[train_i], y[train_i]
    model = DanceTreeClassifier(device=DEVICE)
    for epoch in tqdm(range(1, EPOCHS + 1)):
        # Shuffle the data
        i = np.arange(len(train_paths))
        np.random.shuffle(i)
        train_paths = train_paths[i]
        train_y = train_y[i]
        train_x = features_from_path(train_paths)
        model.fit(train_x, train_y)

    # evaluate the model
    preds = model.predict(x[test_i])
    accuracy = (preds == y[test_i]).mean()
    print(f"{accuracy=}")
    model.save("models/weights/decision_tree")

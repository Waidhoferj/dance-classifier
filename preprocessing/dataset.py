import importlib
import json
import os
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import numpy as np
import pandas as pd
import torchaudio as ta
import pytorch_lightning as pl
from glob import iglob

from preprocessing.preprocess import (
    fix_dance_rating_counts,
    get_unique_labels,
    has_valid_audio,
    url_to_filename,
    vectorize_label_probs,
    vectorize_multi_label,
)


class SongDataset(Dataset):
    def __init__(
        self,
        audio_paths: list[str],
        dance_labels: list[np.ndarray],
        audio_start_offset=6,  # seconds
        audio_window_duration=6,  # seconds
        audio_window_jitter=1.0,  # seconds
        audio_durations=None,
        target_sample_rate=16000,
    ):
        assert (
            audio_window_duration > audio_window_jitter
        ), "Jitter should be a small fraction of the audio window duration."

        self.audio_paths = audio_paths
        self.dance_labels = dance_labels

        # Added to limit file I/O
        if audio_durations is None:
            audio_metadata = [ta.info(audio) for audio in audio_paths]
            self.audio_durations = [
                meta.num_frames / meta.sample_rate for meta in audio_metadata
            ]
            self.sample_rate = audio_metadata[
                0
            ].sample_rate  # assuming same sample rate
        else:
            self.audio_durations = audio_durations
            self.sample_rate = ta.info(
                audio_paths[0]
            ).sample_rate  # assuming same sample rate
        self.audio_window_duration = int(audio_window_duration)
        self.audio_start_offset = audio_start_offset
        self.audio_window_jitter = audio_window_jitter
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return int(
            sum(
                max(duration - self.audio_start_offset, 0) // self.audio_window_duration
                for duration in self.audio_durations
            )
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, list):
            return [
                (self._waveform_from_index(i), self._label_from_index(i)) for i in idx
            ]

        waveform = self._waveform_from_index(idx)
        dance_labels = self._label_from_index(idx)
        return waveform, dance_labels

    def _idx2audio_idx(self, idx: int) -> int:
        return self._get_audio_loc_from_idx(idx)[0]

    def _get_audio_loc_from_idx(self, idx: int) -> tuple[int, int]:
        """
        Converts dataset index to the indices that reference the target audio path
        and window offset.
        """
        total_slices = 0
        for audio_index, duration in enumerate(self.audio_durations):
            audio_slices = max(
                (duration - self.audio_start_offset) // self.audio_window_duration, 1
            )
            if total_slices + audio_slices > idx:
                frame_index = idx - total_slices
                return audio_index, frame_index
            total_slices += audio_slices

    def get_label_weights(self):
        n_examples, n_classes = self.dance_labels.shape
        weights = n_examples / (n_classes * sum(self.dance_labels))
        weights[np.isinf(weights)] = 0.0
        return torch.from_numpy(weights)

    def _backtrace_audio_path(self, index: int) -> str:
        return self.audio_paths[self._idx2audio_idx(index)]

    def _validate_output(self, x, y):
        is_finite = not torch.any(torch.isinf(x))
        is_numerical = not torch.any(torch.isnan(x))
        has_data = torch.any(x != 0.0)
        is_binary = len(torch.unique(y)) < 3
        return all((is_finite, is_numerical, has_data, is_binary))

    def _waveform_from_index(self, idx: int) -> torch.Tensor:
        audio_index, frame_index = self._get_audio_loc_from_idx(idx)
        audio_filepath = self.audio_paths[audio_index]
        num_windows = self.audio_durations[audio_index] // self.audio_window_duration
        jitter_start = -self.audio_window_jitter if frame_index > 0 else 0.0
        jitter_end = self.audio_window_jitter if frame_index != num_windows - 1 else 0.0
        jitter = int(
            torch.FloatTensor(1).uniform_(jitter_start, jitter_end) * self.sample_rate
        )
        frame_offset = int(
            frame_index * self.audio_window_duration * self.sample_rate
            + jitter
            + self.audio_start_offset * self.sample_rate
        )
        num_frames = self.sample_rate * self.audio_window_duration
        waveform, sample_rate = ta.load(
            audio_filepath, frame_offset=frame_offset, num_frames=num_frames
        )
        waveform = ta.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=self.target_sample_rate
        )
        return waveform

    def _label_from_index(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.dance_labels[self._idx2audio_idx(idx)])


class HuggingFaceDatasetWrapper(Dataset):
    """
    Makes a standard PyTorch Dataset compatible with a HuggingFace Trainer.
    """

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.pipeline = []

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x, y = self.dataset[idx]
        if len(self.pipeline) > 0:
            for fn in self.pipeline:
                x = fn(x)

        dance_labels = y.argmax()
        return {
            "input_values": x["input_values"][0] if hasattr(x, "input_values") else x,
            "label": dance_labels,
        }

    def __len__(self):
        return len(self.dataset)

    def append_to_pipeline(self, fn):
        """
        Adds a preprocessing step to the dataset.
        """
        self.pipeline.append(fn)


class BestBallroomDataset(Dataset):
    def __init__(
        self, audio_dir="data/ballroom-songs", class_list=None, **kwargs
    ) -> None:
        super().__init__()
        song_paths, encoded_labels, str_labels = self.get_examples(
            audio_dir, class_list
        )
        self.labels = str_labels
        with open(os.path.join(audio_dir, "audio_durations.json"), "r") as f:
            durations = json.load(f)
            durations = {
                os.path.join(audio_dir, filepath): duration
                for filepath, duration in durations.items()
            }
        audio_durations = [durations[song] for song in song_paths]
        self.song_dataset = SongDataset(
            song_paths, encoded_labels, audio_durations=audio_durations, **kwargs
        )

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.song_dataset[index]

    def __len__(self):
        return len(self.song_dataset)

    def get_examples(self, audio_dir, class_list=None):
        dances = set(
            f
            for f in os.listdir(audio_dir)
            if os.path.isdir(os.path.join(audio_dir, f))
        )
        common_dances = dances
        if class_list is not None:
            common_dances = dances & set(class_list)
            dances = class_list
        dances = np.array(sorted(dances))
        song_paths = []
        labels = []
        for dance in common_dances:
            dance_label = (dances == dance).astype("float32")
            folder_path = os.path.join(audio_dir, dance)
            folder_contents = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
            song_paths.extend(os.path.join(folder_path, f) for f in folder_contents)
            labels.extend([dance_label] * len(folder_contents))

        return np.array(song_paths), np.stack(labels), dances


class Music4DanceDataset(Dataset):
    def __init__(
        self,
        song_data_path,
        song_audio_path,
        class_list=None,
        multi_label=True,
        min_votes=1,
        **kwargs,
    ) -> None:
        super().__init__()
        df = pd.read_csv(song_data_path)
        song_paths, labels = get_music4dance_examples(
            df,
            song_audio_path,
            class_list=class_list,
            multi_label=multi_label,
            min_votes=min_votes,
        )
        self.song_dataset = SongDataset(
            song_paths,
            labels,
            audio_durations=[30.0] * len(song_paths),
            **kwargs,
        )

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.song_dataset[index]

    def __len__(self):
        return len(self.song_dataset)


def get_music4dance_examples(
    df: pd.DataFrame, audio_dir: str, class_list=None, multi_label=True, min_votes=1
) -> tuple[np.ndarray, np.ndarray]:
    sampled_songs = df[has_valid_audio(df["Sample"], audio_dir)].copy(deep=True)
    sampled_songs["DanceRating"] = fix_dance_rating_counts(sampled_songs["DanceRating"])
    if class_list is not None:
        class_list = set(class_list)
        sampled_songs["DanceRating"] = sampled_songs["DanceRating"].apply(
            lambda labels: {k: v for k, v in labels.items() if k in class_list}
            if not pd.isna(labels)
            and any(label in class_list and amt > 0 for label, amt in labels.items())
            else np.nan
        )
    sampled_songs = sampled_songs.dropna(subset=["DanceRating"])
    vote_mask = sampled_songs["DanceRating"].apply(
        lambda dances: any(votes >= min_votes for votes in dances.values())
    )
    sampled_songs = sampled_songs[vote_mask]
    labels = sampled_songs["DanceRating"].apply(
        lambda dances: {
            dance: votes for dance, votes in dances.items() if votes >= min_votes
        }
    )
    unique_labels = np.array(get_unique_labels(labels))
    vectorizer = vectorize_multi_label if multi_label else vectorize_label_probs
    labels = labels.apply(lambda i: vectorizer(i, unique_labels))

    audio_paths = [
        os.path.join(audio_dir, url_to_filename(url)) for url in sampled_songs["Sample"]
    ]

    return np.array(audio_paths), np.stack(labels)


class PipelinedDataset(Dataset):
    """
    Adds a feature extractor preprocessing step to a dataset.
    """

    def __init__(self, dataset, feature_extractor):
        self._data = dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        sample, label = self._data[index]

        features = self.feature_extractor(sample)
        return features, label


class DanceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        test_proportion=0.15,
        val_proportion=0.1,
        target_classes: list[str] = None,
        batch_size: int = 64,
        num_workers=10,
        data_subset=None,
    ):
        super().__init__()
        self.val_proportion = val_proportion
        self.test_proportion = test_proportion
        self.train_proportion = 1.0 - test_proportion - val_proportion
        self.target_classes = target_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        if data_subset is not None and float(data_subset) != 1.0:
            dataset, _ = random_split(dataset, [data_subset, 1 - data_subset])

        self.dataset = dataset

    def setup(self, stage: str):
        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset,
            [self.train_proportion, self.val_proportion, self.test_proportion],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def get_label_weights(self):
        dataset = (
            self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset
        )
        weights = [ds.song_dataset.get_label_weights() for ds in dataset._data.datasets]
        return torch.mean(torch.stack(weights), dim=0)  # TODO: Make this weighted


def find_mean_std(dataset: Dataset, zscore=1.96, moe=0.02, p=0.5):
    """
    Estimates the mean and standard deviations of the a dataset.
    """
    sample_size = int(np.ceil((zscore**2 * p * (1 - p)) / (moe**2)))
    sample_indices = np.random.choice(
        np.arange(len(dataset)), size=sample_size, replace=False
    )
    mean = 0
    std = 0
    for i in sample_indices:
        features = dataset[i][0]
        mean += features.mean().item()
        std += features.std().item()
    print("std", std / sample_size)
    print("mean", mean / sample_size)


def get_datasets(dataset_config: dict, feature_extractor) -> Dataset:
    datasets = []
    for dataset_path, kwargs in dataset_config.items():
        module_name, class_name = dataset_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ProvidedDataset = getattr(module, class_name)
        datasets.append(ProvidedDataset(**kwargs))
    return PipelinedDataset(ConcatDataset(datasets), feature_extractor)


def get_class_counts(config: dict):
    # TODO: Figure out why music4dance has fractional labels
    dataset = get_datasets(config["datasets"], lambda x: x)
    counts = sum(
        np.sum(
            np.arange(len(config["dance_ids"]))
            == np.expand_dims(ds.song_dataset.dance_labels.argmax(1), 1),
            axis=0,
        )
        for ds in dataset._data.datasets
    )
    labels = sorted(config["dance_ids"])
    return dict(zip(labels, counts))


def record_audio_durations(folder: str):
    """
    Records a filename: duration mapping of all audio files in a folder to a json file.
    """
    durations = {}
    music_files = iglob(os.path.join(folder, "**", "*.wav"), recursive=True)
    for file in music_files:
        meta = ta.info(file)
        durations[file] = meta.num_frames / meta.sample_rate

    with open(os.path.join(folder, "audio_durations.json"), "w") as f:
        json.dump(durations, f)

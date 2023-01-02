import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import torchaudio as ta
from .pipelines import AudioPipeline
import pytorch_lightning as pl
from .preprocess import get_examples



class SongDataset(Dataset):
    def __init__(self, 
    audio_paths: list[str], 
    dance_labels: list[np.ndarray], 
    audio_duration=30, # seconds
    audio_window_duration=6, # seconds
    ):
        assert audio_duration % audio_window_duration == 0, "Audio window should divide duration evenly."

        self.audio_paths = audio_paths
        self.dance_labels = dance_labels
        audio_info = ta.info(audio_paths[0])
        self.sample_rate = audio_info.sample_rate
        self.audio_window_duration = int(audio_window_duration)
        self.audio_duration = int(audio_duration)

        self.audio_pipeline = AudioPipeline(input_freq=self.sample_rate)

    def __len__(self):
        return len(self.audio_paths) * self.audio_duration // self.audio_window_duration 

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self._waveform_from_index(idx)
        spectrogram = self.audio_pipeline(waveform)

        dance_labels = self._label_from_index(idx)

        return spectrogram, dance_labels


    def _waveform_from_index(self, idx:int) -> torch.Tensor:
        audio_file_idx = idx * self.audio_window_duration // self.audio_duration
        frame_offset = idx % self.audio_duration // self.audio_window_duration
        num_frames = self.sample_rate * self.audio_window_duration
        waveform, sample_rate = ta.load(self.audio_paths[audio_file_idx], frame_offset=frame_offset, num_frames=num_frames)
        assert sample_rate == self.sample_rate, f"Expected sample rate of {self.sample_rate}. Found {sample_rate}"        
        return waveform


    def _label_from_index(self, idx:int) -> torch.Tensor:
        label_idx =  idx * self.audio_window_duration // self.audio_duration
        return torch.from_numpy(self.dance_labels[label_idx])


class DanceDataModule(pl.LightningDataModule):
    def __init__(self, 
    song_data_path="data/songs.csv",
    song_audio_path="data/samples",
    test_proportion=0.15,
    val_proportion=0.1,
    target_classes:list[str]=None,
    batch_size:int=64,
    num_workers=10
    ):
        super().__init__()
        self.song_data_path = song_data_path
        self.song_audio_path = song_audio_path
        self.val_proportion=val_proportion
        self.test_proportion=test_proportion
        self.train_proporition= 1.-test_proportion-val_proportion
        self.target_classes=target_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        df = pd.read_csv("data/songs.csv")
        self.x,self.y = get_examples(df, self.song_audio_path,class_list=self.target_classes)


    def setup(self, stage: str):
        dataset = SongDataset(self.x,self.y)
        self.train_ds, self.val_ds, self.test_ds = random_split(dataset, [self.train_proporition, self.val_proportion, self.test_proportion])
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_label_weights(self):
        return torch.from_numpy(len(self.y) / (len(self.y[0]) * sum(self.y)))
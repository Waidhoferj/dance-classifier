import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import torchaudio as ta
from .pipelines import AudioTrainingPipeline
import pytorch_lightning as pl
from .preprocess import get_examples
from sklearn.model_selection import train_test_split



class SongDataset(Dataset):
    def __init__(self, 
    audio_paths: list[str], 
    dance_labels: list[np.ndarray],
    audio_duration=30, # seconds
    audio_window_duration=6, # seconds
    audio_window_jitter=0.0, # seconds
    audio_pipeline_kwargs={},
    resample_frequency=16000
    ):
        assert audio_duration % audio_window_duration == 0, "Audio window should divide duration evenly."
        assert audio_window_duration > audio_window_jitter, "Jitter should be a small fraction of the audio window duration."

        self.audio_paths = audio_paths
        self.dance_labels = dance_labels
        audio_info = ta.info(audio_paths[0])
        self.sample_rate = audio_info.sample_rate
        self.audio_window_duration = int(audio_window_duration)
        self.audio_window_jitter = audio_window_jitter
        self.audio_duration = int(audio_duration)

        self.audio_pipeline = AudioTrainingPipeline(self.sample_rate, resample_frequency, audio_window_duration, **audio_pipeline_kwargs)

    def __len__(self):
        return len(self.audio_paths) * self.audio_duration // self.audio_window_duration 

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self._waveform_from_index(idx)
        assert waveform.shape[1] > 10, f"No data found: {self._backtrace_audio_path(idx)}"
        spectrogram = self.audio_pipeline(waveform)

        dance_labels = self._label_from_index(idx)

        example_is_valid = self._validate_output(spectrogram, dance_labels)
        if example_is_valid:
            return spectrogram, dance_labels
        else:
            # Try the previous one
            # This happens when some of the audio recordings are really quiet
            # This WILL NOT leak into other data partitions because songs belong entirely to a partition
            return self[idx-1]

    def _convert_idx(self,idx:int) -> int:
        return idx * self.audio_window_duration // self.audio_duration

    def _backtrace_audio_path(self, index:int) -> str:
        return self.audio_paths[self._convert_idx(index)]

    def _validate_output(self,x,y):
        is_finite =  not torch.any(torch.isinf(x))
        is_numerical = not torch.any(torch.isnan(x))
        has_data = torch.any(x != 0.0)
        is_binary = len(torch.unique(y)) < 3
        return all((is_finite,is_numerical, has_data, is_binary))

    def _waveform_from_index(self, idx:int) -> torch.Tensor:
        audio_filepath = self.audio_paths[self._convert_idx(idx)]
        num_windows = self.audio_duration // self.audio_window_duration
        frame_index = idx % num_windows
        jitter_start = -self.audio_window_jitter if frame_index > 0 else 0.0
        jitter_end = self.audio_window_jitter if frame_index != num_windows - 1 else 0.0
        jitter = int(torch.FloatTensor(1).uniform_(jitter_start, jitter_end) * self.sample_rate)
        frame_offset = frame_index * self.audio_window_duration * self.sample_rate + jitter
        num_frames = self.sample_rate * self.audio_window_duration
        waveform, sample_rate = ta.load(audio_filepath, frame_offset=frame_offset, num_frames=num_frames)
        assert sample_rate == self.sample_rate, f"Expected sample rate of {self.sample_rate}. Found {sample_rate}"        
        return waveform


    def _label_from_index(self, idx:int) -> torch.Tensor:
        return torch.from_numpy(self.dance_labels[self._convert_idx(idx)])

class DanceDataModule(pl.LightningDataModule):
    def __init__(self, 
    song_data_path="data/songs_cleaned.csv",
    song_audio_path="data/samples",
    test_proportion=0.15,
    val_proportion=0.1,
    target_classes:list[str]=None,
    min_votes=1,
    batch_size:int=64,
    num_workers=10,
    dataset_kwargs={}
    ):
        super().__init__()
        self.song_data_path = song_data_path
        self.song_audio_path = song_audio_path
        self.val_proportion=val_proportion
        self.test_proportion=test_proportion
        self.train_proportion= 1.-test_proportion-val_proportion
        self.target_classes=target_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs

        df = pd.read_csv(song_data_path)
        self.x,self.y = get_examples(df, self.song_audio_path,class_list=self.target_classes, multi_label=True, min_votes=min_votes)

    def setup(self, stage: str):
        train_i, val_i, test_i = random_split(np.arange(len(self.x)), [self.train_proportion, self.val_proportion, self.test_proportion])
        self.train_ds = self._dataset_from_indices(train_i)
        self.val_ds = self._dataset_from_indices(val_i)
        self.test_ds = self._dataset_from_indices(test_i)
    
    def _dataset_from_indices(self, idx:list[int]) -> SongDataset:
        return SongDataset(self.x[idx], self.y[idx], **self.dataset_kwargs)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_label_weights(self):
        n_examples, n_classes = self.y.shape
        return torch.from_numpy(n_examples / (n_classes * sum(self.y)))
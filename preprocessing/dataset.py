import torch
from torch.utils.data import Dataset
import numpy as np
import torchaudio as ta
from .preprocess import AudioPipeline


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

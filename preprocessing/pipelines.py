import torch
from torchaudio import transforms as taT, functional as taF
import torch.nn as nn

class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        resample_freq=16000,
    ):
        super().__init__()
        self.resample = taT.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spec = taT.MelSpectrogram(sample_rate=resample_freq, n_mels=64, n_fft=1024)
        self.to_db = taT.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        waveform = (waveform - waveform.mean()) / waveform.abs().max()
        
        waveform = self.resample(waveform)
        spectrogram = self.spec(waveform)
        spectrogram = self.to_db(spectrogram)

        return spectrogram


class SpectrogramAugmentationPipeline(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pipeline = nn.Sequential(
            taT.FrequencyMasking(80),
            taT.TimeMasking(80),
            taT.TimeStretch(80)
        )

    def forward(self, spectrogram:torch.Tensor) -> torch.Tensor:
        return self.pipeline(spectrogram)


class WaveformAugmentationPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        


    def forward(self, waveform:torch.Tensor) -> torch.Tensor:
        taF.pitch_shift()


class AudioTrainingPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.waveform_aug = WaveformAugmentationPipeline()
        self.spec_aug = SpectrogramAugmentationPipeline()
        self.audio_preprocessing = AudioPipeline()

    def forward(self, waveform:torch.Tensor) -> torch.Tensor:
        x = self.audio_preprocessing(waveform)
        x = self.spec_aug(x)
        return x 
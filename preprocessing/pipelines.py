import torch
import torchaudio
from torchaudio import transforms as taT, functional as taF
import torch.nn as nn

NOISE_PATH = "data/augmentation/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"

class AudioTrainingPipeline(torch.nn.Module):
    def __init__(self, 
            input_freq=16000,
            resample_freq=16000,
            expected_duration=6,
            freq_mask_size=10,
            time_mask_size=80,
            mask_count = 2,
            snr_mean=6.0):
        super().__init__()
        self.input_freq = input_freq
        self.snr_mean = snr_mean
        self.mask_count = mask_count
        self.noise = self.get_noise()
        self.resample = taT.Resample(input_freq,resample_freq)
        self.preprocess_waveform = WaveformPreprocessing(resample_freq * expected_duration)
        self.audio_to_spectrogram = AudioToSpectrogram(
            sample_rate=resample_freq,
        )
        self.freq_mask = taT.FrequencyMasking(freq_mask_size)
        self.time_mask = taT.TimeMasking(time_mask_size)
        

    def get_noise(self) -> torch.Tensor:
        noise, sr = torchaudio.load(NOISE_PATH)
        if noise.shape[0] > 1:
            noise = noise.mean(0, keepdim=True)
        if sr != self.input_freq:
            noise = taF.resample(noise,sr, self.input_freq)
        return noise

    def add_noise(self, waveform:torch.Tensor) -> torch.Tensor:
        num_repeats = waveform.shape[1] // self.noise.shape[1] + 1
        noise = self.noise.repeat(1,num_repeats)[:, :waveform.shape[1]]
        noise_power = noise.norm(p=2)
        signal_power = waveform.norm(p=2)
        snr_db = torch.normal(self.snr_mean, 1.5, (1,)).clamp_min(1.0)
        snr = torch.exp(snr_db / 10)
        scale = snr * noise_power / signal_power
        noisy_waveform = (scale * waveform + noise) / 2
        return noisy_waveform

    def forward(self, waveform:torch.Tensor) -> torch.Tensor:
        try:
            waveform = self.resample(waveform)
        except:
            print("oops")
        waveform = self.preprocess_waveform(waveform)
        waveform = self.add_noise(waveform)
        spec = self.audio_to_spectrogram(waveform)

        # Spectrogram augmentation
        for _ in range(self.mask_count):
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        return spec


class WaveformPreprocessing(torch.nn.Module):

    def __init__(self, expected_sample_length:int):
        super().__init__()
        self.expected_sample_length = expected_sample_length
        


    def forward(self, waveform:torch.Tensor) -> torch.Tensor:
        # Take out extra channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # ensure it is the correct length
        waveform = self._rectify_duration(waveform)
        return waveform


    def _rectify_duration(self,waveform:torch.Tensor):
        expected_samples = self.expected_sample_length
        sample_count = waveform.shape[1]
        if expected_samples == sample_count:
            return waveform
        elif expected_samples > sample_count:
            pad_amount = expected_samples - sample_count
            return torch.nn.functional.pad(waveform, (0, pad_amount),mode="constant", value=0.0)
        else:
            return waveform[:,:expected_samples]


class AudioToSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000,
    ):
        super().__init__()

        self.spec = taT.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=1024) # TODO: Change mels to 64
        self.to_db = taT.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.spec(waveform)
        spectrogram = self.to_db(spectrogram)
        return spectrogram
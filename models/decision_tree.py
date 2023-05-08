
from sklearn.base import ClassifierMixin, BaseEstimator
import pandas as pd
from torch import nn
import torch
from typing import Iterator
import numpy as np
import json
from tqdm import tqdm
import librosa
DANCE_INFO_FILE = "data/dance_info.csv"
dance_info_df = pd.read_csv(DANCE_INFO_FILE, converters={'tempoRange': lambda s : json.loads(s.replace("'", '"'))})

class DanceTreeClassifier(BaseEstimator,ClassifierMixin):
    """
    Trains a series of binary classifiers to classify each dance when a song falls into its bpm range.

    Features:
        - Spectrogram
        - BPM
    """

    def __init__(self, device="cpu", lr=1e-4, epochs=5, verbose=True) -> None:
        self.device=device
        self.epochs=epochs
        self.verbose = verbose
        self.lr = lr
        self.classifiers = {}
        self.optimizers = {}
        self.criterion = nn.BCELoss()

    def get_valid_dances_from_bpm(self,bpm:float) -> list[str]:
        mask = dance_info_df["tempoRange"].apply(lambda interval: interval["min"] <= bpm <= interval["max"])
        return list(dance_info_df["id"][mask])

        

    def fit(self, x, y):
        """
        x: (specs, bpms). The first element is the spectrogram, second element is the bpm. spec shape should be (channel, freq_bins, sr * time)
        y: (batch_size, n_classes)
        """
        progress_bar = tqdm(range(self.epochs))
        for _ in progress_bar:
            # TODO: Introduce batches
            epoch_loss = 0
            pred_count = 0
            for (spec, bpm), label in zip(x, y):
                # find all models that are in the bpm range
                matching_dances = self.get_valid_dances_from_bpm(bpm)
                for dance in matching_dances:
                    if dance not in self.classifiers or dance not in self.optimizers:
                        classifier = DanceCNN()
                        self.classifiers[dance] = classifier
                        self.optimizers[dance] = torch.optim.Adam(classifier.parameters(), lr=self.lr)
                models = [(dance, model, self.optimizers[dance]) for dance, model in self.classifiers.items() if dance in matching_dances]
                for dance, model,opt in models:
                    opt.zero_grad()
                    spec = torch.from_numpy(spec).to(self.device)
                    output = model(spec)
                    target = torch.tensor(float(dance == label))
                    loss = self.criterion(output, target)
                    epoch_loss += loss.item()
                    pred_count +=1
                    loss.backward()
                    opt.step()
            progress_bar.set_description(f"Loss: {epoch_loss / pred_count}")

    def predict(self, x) -> list[str]:
        results = []
        for spec, bpm in zip(*x):
            matching_dances = self.get_valid_dances_from_bpm(bpm)
            dance_i = torch.tensor([self.classifiers[dance](spec) for dance in matching_dances]).argmax()
            results.append(matching_dances[dance_i])
        return results

        


class DanceCNN(nn.Module):
    def __init__(self, sr=16000, freq_bins=20, duration=6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        kernel_size=(3,9)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2,10)),
            nn.Conv2d(16,32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2,10))
        )

        embedding_dimension = 32* 3 * 959
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten() if len(x.shape) == 3 else x.flatten(1)
        return self.classifier(x)

def features_from_path(paths:list[str], 
                       audio_window_duration=6, 
                       audio_duration=30,
                       resample_freq=16000) -> Iterator[tuple[np.array, float]]:
    """
    Loads audio and bpm from an audio path.
    """
    
    for path in paths:
        waveform, sr = librosa.load(path, mono=True, sr=resample_freq)
        num_frames =  audio_window_duration * sr
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
        mfccs_normalized = (mfccs - mfccs.mean()) / mfccs.std()
        mfccs_padded = librosa.util.fix_length(mfccs_normalized, size=sr*audio_duration, axis=1)
        mfccs_reshaped = mfccs_padded.reshape(1, mfccs_padded.shape[0], mfccs_padded.shape[1])
        for i in range(audio_duration//audio_window_duration):
            mfcc_window = mfccs_reshaped[:,:,i*num_frames:(i+1)*num_frames] 
            yield (mfcc_window, tempo)

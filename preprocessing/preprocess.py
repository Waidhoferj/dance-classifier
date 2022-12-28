import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import os
import torch
import torchaudio.transforms as taT

def url_to_filename(url:str) -> str:
    return f"{url.split('/')[-1]}.wav"

def get_songs_with_audio(df:pd.DataFrame, audio_dir:str) -> pd.DataFrame:
    audio_urls = df["Sample"].replace(".", np.nan)
    audio_files = set(os.path.basename(f) for f in Path(audio_dir).iterdir())
    valid_audio = audio_urls.apply(lambda url : url is not np.nan and url_to_filename(url) in audio_files)
    df = df[valid_audio]
    return df

def fix_dance_rating_counts(dance_ratings:pd.Series) -> pd.Series:
    tag_pattern = re.compile("([A-Za-z]+)(\+|-)(\d+)")
    dance_ratings = dance_ratings.apply(lambda v : json.loads(v.replace("'", "\"")))
    def fix_labels(labels:dict) -> dict | float:
        new_labels = {}
        for k, v in labels.items():
            match = tag_pattern.search(k)
            if match is None:
                new_labels[k] = new_labels.get(k, 0) + v
            else:
                k = match[1]
                sign = 1 if match[2] == '+' else -1
                scale = int(match[3])
                new_labels[k] = new_labels.get(k, 0) + v * scale * sign
        valid = any(v > 0 for v in new_labels.values())
        return new_labels if valid else np.nan
    return dance_ratings.apply(fix_labels)


def get_unique_labels(dance_labels:pd.Series) -> list:
    labels = set()
    for dances in dance_labels:
        labels |= set(dances)
    return sorted(labels)

def vectorize_label_probs(labels: dict[str,int], unique_labels:np.ndarray) -> np.ndarray:
    """
    Turns label dict into probability distribution vector based on each label count.
    """
    label_vec = np.zeros((len(unique_labels),), dtype="float32")
    for k, v in labels.items():
        item_vec = (unique_labels == k) * v
        label_vec += item_vec
    lv_cache = label_vec.copy()
    label_vec[label_vec<0] = 0
    label_vec /= label_vec.sum()
    assert not any(np.isnan(label_vec)), f"Provided labels are invalid: {labels}"
    return label_vec

def vectorize_multi_label(labels: dict[str,int], unique_labels:np.ndarray) -> np.ndarray:
    """
    Turns label dict into binary label vectors for multi-label classification.
    """
    probs = vectorize_label_probs(labels,unique_labels)
    probs[probs > 0.0] = 1.0
    return probs

def get_examples(df:pd.DataFrame, audio_dir:str, class_list=None) -> tuple[list[str], list[np.ndarray]]:
    sampled_songs = get_songs_with_audio(df, audio_dir)
    sampled_songs.loc[:,"DanceRating"] = fix_dance_rating_counts(sampled_songs["DanceRating"])
    if class_list is not None:
        class_list = set(class_list)
        sampled_songs.loc[:,"DanceRating"] = sampled_songs["DanceRating"].apply(
            lambda labels : {k: v for k,v in labels.items() if k in class_list} 
            if not pd.isna(labels) and any(label in class_list and amt > 0 for label, amt in labels.items()) 
            else np.nan)
    sampled_songs = sampled_songs.dropna(subset=["DanceRating"])
    labels = sampled_songs["DanceRating"]
    unique_labels = np.array(get_unique_labels(labels))
    labels = labels.apply(lambda i : vectorize_multi_label(i, unique_labels))

    audio_paths = [os.path.join(audio_dir, url_to_filename(url)) for url in sampled_songs["Sample"]]

    return audio_paths, list(labels)

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
        waveform = self.resample(waveform)
        spectrogram = self.spec(waveform)
        spectrogram = self.to_db(spectrogram)

        return spectrogram


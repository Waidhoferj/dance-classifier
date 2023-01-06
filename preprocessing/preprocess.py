import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import os
import torchaudio
import torch
from tqdm import tqdm

def url_to_filename(url:str) -> str:
    return f"{url.split('/')[-1]}.wav"

def get_songs_with_audio(df:pd.DataFrame, audio_dir:str) -> pd.DataFrame:
    audio_urls = df["Sample"].replace(".", np.nan)
    audio_files = set(os.path.basename(f) for f in Path(audio_dir).iterdir())
    valid_audio = audio_urls.apply(lambda url : url is not np.nan and url_to_filename(url) in audio_files)
    df = df[valid_audio]
    return df

def validate_audio(audio_urls:pd.Series, audio_dir:str) -> pd.Series:
    """
    Tests audio urls to ensure that their file exists and the contents is valid. 
    """
    audio_files = set(os.path.basename(f) for f in Path(audio_dir).iterdir())
    def is_valid(url):
        valid_url = type(url) == str and "http" in url 
        if not valid_url:
            return False
        filename = url_to_filename(url)
        if filename not in audio_files:
            return False
        try:
            w, _ = torchaudio.load(os.path.join(audio_dir, filename))
        except:
            return False
        contents_invalid = torch.any(torch.isnan(w)) or torch.any(torch.isinf(w)) or len(torch.unique(w)) <= 2
        return not contents_invalid
    
    idxs = []
    validations = []
    for index, url in tqdm(audio_urls.items(), total=len(audio_urls), desc="Audio URLs Validated"):
        idxs.append(index)
        validations.append(is_valid(url))

    return pd.Series(validations, index=idxs)
    
    

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

def get_examples(df:pd.DataFrame, audio_dir:str, class_list=None, multi_label=True, min_votes=1) -> tuple[np.ndarray, np.ndarray]:
    sampled_songs = get_songs_with_audio(df, audio_dir)
    sampled_songs.loc[:,"DanceRating"] = fix_dance_rating_counts(sampled_songs["DanceRating"])
    if class_list is not None:
        class_list = set(class_list)
        sampled_songs.loc[:,"DanceRating"] = sampled_songs["DanceRating"].apply(
            lambda labels : {k: v for k,v in labels.items() if k in class_list} 
            if not pd.isna(labels) and any(label in class_list and amt > 0 for label, amt in labels.items()) 
            else np.nan)
    sampled_songs = sampled_songs.dropna(subset=["DanceRating"])
    vote_mask = sampled_songs["DanceRating"].apply(lambda dances: any(votes >= min_votes for votes in dances.values()))
    sampled_songs = sampled_songs[vote_mask]
    labels = sampled_songs["DanceRating"].apply(lambda dances : {dance: votes for dance, votes in dances.items() if votes >= min_votes})
    unique_labels = np.array(get_unique_labels(labels))
    vectorizer = vectorize_multi_label if multi_label else vectorize_label_probs
    labels = labels.apply(lambda i : vectorizer(i, unique_labels))

    audio_paths = [os.path.join(audio_dir, url_to_filename(url)) for url in sampled_songs["Sample"]]

    return np.array(audio_paths), np.stack(labels)


if __name__ == "__main__":
    links = pd.read_csv("data/backup_2.csv", index_col="index")
    df = pd.read_csv("data/songs.csv")
    l = links["link"].str.strip()
    l = l.apply(lambda url : url if "http" in url else np.nan)
    l = l.dropna()
    df["Sample"].update(l)
    addna = lambda url :  url if type(url) == str and "http" in url else np.nan
    df["Sample"] = df["Sample"].apply(addna)
    is_valid = validate_audio(df["Sample"],"data/samples")
    df["valid"] = is_valid
    df.to_csv("data/songs_validated.csv")


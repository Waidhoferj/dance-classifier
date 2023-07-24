import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import glob
import os
import shutil
import torchaudio
import torch
from tqdm import tqdm

from preprocessing.utils import url_to_filename


def has_valid_audio(audio_urls: pd.Series, audio_dir: str) -> pd.Series:
    audio_urls = audio_urls.replace(".", np.nan)
    audio_files = set(os.path.basename(f) for f in Path(audio_dir).iterdir())
    valid_audio_mask = audio_urls.apply(
        lambda url: url is not np.nan and url_to_filename(url) in audio_files
    )
    return valid_audio_mask


def validate_audio(audio_urls: pd.Series, audio_dir: str) -> pd.Series:
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
        contents_invalid = (
            torch.any(torch.isnan(w))
            or torch.any(torch.isinf(w))
            or len(torch.unique(w)) <= 2
        )
        return not contents_invalid

    idxs = []
    validations = []
    for index, url in tqdm(
        audio_urls.items(), total=len(audio_urls), desc="Audio URLs Validated"
    ):
        idxs.append(index)
        validations.append(is_valid(url))

    return pd.Series(validations, index=idxs)


def fix_dance_rating_counts(dance_ratings: pd.Series) -> pd.Series:
    tag_pattern = re.compile("([A-Za-z]+)(\+|-)(\d+)")
    dance_ratings = dance_ratings.apply(lambda v: json.loads(v.replace("'", '"')))

    def fix_labels(labels: dict) -> dict | float:
        new_labels = {}
        for k, v in labels.items():
            match = tag_pattern.search(k)
            if match is None:
                new_labels[k] = new_labels.get(k, 0) + v
            else:
                k = match[1]
                sign = 1 if match[2] == "+" else -1
                scale = int(match[3])
                new_labels[k] = new_labels.get(k, 0) + v * scale * sign
        valid = any(v > 0 for v in new_labels.values())
        return new_labels if valid else np.nan

    return dance_ratings.apply(fix_labels)


def get_unique_labels(dance_labels: pd.Series) -> list:
    labels = set()
    for dances in dance_labels:
        labels |= set(dances)
    return sorted(labels)


def vectorize_label_probs(
    labels: dict[str, int], unique_labels: np.ndarray
) -> np.ndarray:
    """
    Turns label dict into probability distribution vector based on each label count.
    """
    label_vec = np.zeros((len(unique_labels),), dtype="float32")
    for k, v in labels.items():
        item_vec = (unique_labels == k) * v
        label_vec += item_vec
    label_vec[label_vec < 0] = 0
    label_vec /= label_vec.sum()
    assert not any(np.isnan(label_vec)), f"Provided labels are invalid: {labels}"
    return label_vec


def vectorize_multi_label(
    labels: dict[str, int], unique_labels: np.ndarray
) -> np.ndarray:
    """
    Turns label dict into binary label vectors for multi-label classification.
    """
    probs = vectorize_label_probs(labels, unique_labels)
    probs[probs > 0.0] = 1.0
    return probs


def sort_yt_files(
    aliases_path="data/dance_aliases.json",
    all_dances_folder="data/best-ballroom-music",
    original_location="data/yt-ballroom-music/",
):
    def normalize_string(s):
        # Lowercase string and remove special characters
        return re.sub(r"\W+", "", s.lower())

    with open(aliases_path, "r") as f:
        dances = json.load(f)

    # Normalize the dance inputs and aliases
    normalized_dances = {
        normalize_string(dance_id): [normalize_string(alias) for alias in aliases]
        for dance_id, aliases in dances.items()
    }

    # For every wav file in the target folder
    bad_files = []
    progress_bar = tqdm(os.listdir(all_dances_folder), unit="files moved")
    for file_name in progress_bar:
        if file_name.endswith(".wav"):
            # check if the normalized wav file name contains the normalized dance alias
            normalized_file_name = normalize_string(file_name)

            matching_dance_ids = [
                dance_id
                for dance_id, aliases in normalized_dances.items()
                if any(alias in normalized_file_name for alias in aliases)
            ]

            if len(matching_dance_ids) == 0:
                # See if the dance is in the path
                original_filename = file_name.replace(".wav", "")
                matches = glob.glob(
                    os.path.join(original_location, "**", original_filename),
                    recursive=True,
                )
                if len(matches) == 1:
                    normalized_file_name = normalize_string(matches[0])
                    matching_dance_ids = [
                        dance_id
                        for dance_id, aliases in normalized_dances.items()
                        if any(alias in normalized_file_name for alias in aliases)
                    ]

            if "swz" in matching_dance_ids and "vwz" in matching_dance_ids:
                matching_dance_ids.remove("swz")
            if len(matching_dance_ids) > 1 and "lhp" in matching_dance_ids:
                matching_dance_ids.remove("lhp")

            if len(matching_dance_ids) != 1:
                bad_files.append(file_name)
                progress_bar.set_description(f"bad files: {len(bad_files)}")
                continue
            dst = os.path.join("data", "ballroom-songs", matching_dance_ids[0].upper())
            os.makedirs(dst, exist_ok=True)
            filepath = os.path.join(all_dances_folder, file_name)
            shutil.copy(filepath, os.path.join(dst, file_name))

    with open("data/bad_files.json", "w") as f:
        json.dump(bad_files, f)


if __name__ == "__main__":
    sort_yt_files()

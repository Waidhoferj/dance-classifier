import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import download_song
import time

def set_env():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "auth", "spotify.json"), "r") as f:
        config = json.load(f)
    os.environ["SPOTIPY_CLIENT_ID"] = config["client_id"]
    os.environ["SPOTIPY_CLIENT_SECRET"] = config["client_secret"]
    os.environ["SPOTIPY_REDIRECT_URI"] = "https://localhost:8080/callback"

set_env()


def get_song_preview_url(song_name:str, spotify:spotipy.Spotify, artist:str = None) -> str | None:
    info = {
        "track": song_name
    }
    if artist is not None:
        info["artist"] = artist
    query = " ".join(f"{k}: {v}" for k,v in info.items())
    results = spotify.search(query,type="track", limit=1)["tracks"]["items"]
    valid_results = len(results) > 0 and results[0] is not None and "preview_url" in results[0]
    if not valid_results:
        return None
    song = results[0]
    return song["preview_url"]

def patch_missing_songs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
    # find songs with missing previews
    audio_urls = df["Sample"].replace(".", np.nan)
    missing_audio = pd.isna(audio_urls)
    missing_df = df[missing_audio]
    def patch_preview(row: pd.Series):
        song:str = row["Title"]
        artist:str = row["Artist"]
        preview_url = get_song_preview_url(song, spotify, artist)
        if preview_url is not None:
            row["Sample"] = preview_url
        return row
    rows = []
    indices = []
    after = 18418
    missing_df = missing_df.iloc[after:]
    total_rows = len(missing_df)
    for i, row in tqdm(missing_df.iterrows(),total=total_rows):
        patched_row = patch_preview(row)
        rows.append(patched_row)
        indices.append(i)


    patched_df = pd.DataFrame(rows,index=indices)
    df.update(patched_df)
    return df


def download_links_from_backup(backup_file:str, output_dir:str):
    with open(backup_file) as f:
        links = [x.split(",")[1].strip() for x in f.readlines()]
    links = [l for l in links if "https" in l]
    for link in tqdm(links, "Songs Downloaded"):
        download_song(link, output_dir)
        time.sleep(5e-3) # hopefully wont be rate limited with delay 🤞

import requests
from bs4 import BeautifulSoup as bs
import json
import argparse
from pathlib import Path
import os
import pandas as pd
import re
from tqdm import tqdm




def scrape_song_library(page_count=2054) -> pd.DataFrame:
    columns = [
        "Title",
        "Artist",
        "Length",
        "Tempo",
        "Beat",
        "Energy",
        "Danceability",
        "Valence",
        "Sample",
        "Tags",
        "DanceRating",
    ]
    song_df = pd.DataFrame(columns=columns)
    for i in tqdm(range(1, page_count + 1), desc="Pages processed"):
        link = "https://www.music4dance.net/song/Index?filter=v2-Index&page=" + str(i)
        page = requests.get(link)
        soup = bs(page.content, "html.parser")
        songs = pd.DataFrame(get_songs(soup))
        song_df = pd.concat([song_df, songs], axis=0, ignore_index=True)
    return song_df


def get_songs(soup: bs) -> dict:
    js_obj = re.compile(r"{(.|\n)*}")
    reset_keys = [
        "Title",
        "Artist",
        "Length",
        "Tempo",
        "Beat",
        "Energy",
        "Danceability",
        "Valence",
        "Sample",
    ]
    song_text = [str(v) for v in soup.find_all("script") if "histories" in str(v)][0]
    songs_data = json.loads(js_obj.search(song_text).group(0))
    songs = []
    for song_data in songs_data["histories"]:
        song = {"Tags": set(), "DanceRating": {}}
        for feature in song_data["properties"]:
            if "name" not in feature or "value" not in feature:
                continue
            key = feature["name"]
            value = feature["value"]
            if key in reset_keys:
                song[key] = value
            elif key == "Tag+":
                song["Tags"].add(value)
            elif key == "DeleteTag":
                try:
                    song["Tags"].remove(value)
                except:
                    continue
            elif key == "DanceRating":
                dance = value.replace("+1", "")
                prev = song["DanceRating"].get(dance, 0)
                song["DanceRating"][dance] = prev + 1
        songs.append(song)
    return songs



def scrape_dance_info() -> pd.DataFrame:
    js_obj = re.compile(r"{(.|\n)*}")
    link = "https://www.music4dance.net/song/Index?filter=v2-Index"
    page = requests.get(link)
    soup = bs(page.content, "html.parser")

    dance_info_text = [str(v) for v in soup.find_all("script") if "environment" in str(v)][0]
    dance_info = json.loads(js_obj.search(dance_info_text).group(0))
    dance_info = dance_info["dances"]
    wanted_keys = ["name", "id", "synonyms", "tempoRange", "songCount"]
    dance_df = pd.DataFrame([{k:v for k, v in dance.items() if k in wanted_keys}
     for dance 
     in dance_info])
    return dance_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-count", default=2, type=int)
    parser.add_argument("--out", default="data/song.csv")

    args = parser.parse_args()
    out_path = Path(args.out)
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        print(f"Output location does not exist: {out_dir}")
    df = scrape_song_library(args.page_count)
    df.to_csv(out_path)

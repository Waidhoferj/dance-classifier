import requests
from pathlib import Path

def download_song(url: str, out_dir: str, file_type="mp3"):
    response = requests.get(url)
    filename = url.split("/")[-1]
    out_file = Path(out_dir, f"{filename}.{file_type}")
    with open(out_file, "wb") as f:
        f.write(response.content)
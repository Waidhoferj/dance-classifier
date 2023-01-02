from pathlib import Path
import gradio as gr
import numpy as np
from models.residual import DancePredictor
import os
from functools import cache
from pathlib import Path
CONFIG_FILE = Path("models/config/dance-predictor.yaml")


@cache
def get_model(config_path:str) -> DancePredictor:
    model = DancePredictor.from_config(config_path)
    return model

def predict(audio: tuple[int, np.ndarray]) -> list[str]:
    sample_rate, waveform = audio
    
    model = get_model(CONFIG_FILE)
    results = model(waveform,sample_rate)
    return results if len(results) else "Dance Not Found"


def demo():
    title = "Dance Classifier"
    description = "What should I dance to this song? Pass some audio to the Dance Classifier find out!"
    song_samples = Path(os.path.dirname(__file__), "assets", "song-samples")
    example_audio = [str(song) for song in song_samples.iterdir() if song.name[0] != '.']
    all_dances = get_model(CONFIG_FILE).labels
    
    recording_interface = gr.Interface(
        fn=predict,
        description="Record at least **6 seconds** of the song.",
        inputs=gr.Audio(source="microphone", label="Song Recording"),
        outputs=gr.Label(label="Dances"),
        examples=example_audio
    )
    uploading_interface = gr.Interface(
        fn=predict,
        inputs=gr.Audio(label="Song Audio File"),
        outputs=gr.Label(label="Dances"),
        examples=example_audio
    )
    
    with gr.Blocks() as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        gr.TabbedInterface([uploading_interface, recording_interface], ["Upload Song", "Record Song"])
        with gr.Accordion("See all dances", open=False):
            gr.Markdown("\n".join(f"- {dance}" for dance in all_dances))

    

    return app


if __name__ == "__main__":
    demo().launch()
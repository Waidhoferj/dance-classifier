from pathlib import Path
import gradio as gr
import numpy as np
import torch
from preprocessing.preprocess import AudioPipeline
from models.residual import ResidualDancer
import os
import json
from functools import cache
import pandas as pd

DEVICE = "cpu"

@cache
def get_model(device) -> tuple[ResidualDancer, np.ndarray]:
    model_path = "models/weights/ResidualDancer"
    weights = os.path.join(model_path, "dancer_net.pt")
    config_path = os.path.join(model_path, "config.json")

    with open(config_path) as f:
        config = json.load(f)
    labels = np.array(sorted(config["classes"]))

    model = ResidualDancer(n_classes=len(labels))
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model = model.to(device).eval()
    return model, labels

@cache
def get_pipeline(sample_rate:int) -> AudioPipeline:
    return AudioPipeline(input_freq=sample_rate)

@cache
def get_dance_map() -> dict:
    df = pd.read_csv("data/dance_mapping.csv")
    return df.set_index("id").to_dict()["name"]


def predict(audio: tuple[int, np.ndarray]) -> list[str]:
    sample_rate, waveform = audio
    
    expected_duration = 6
    threshold = 0.5
    sample_len = sample_rate * expected_duration
    

    audio_pipeline = get_pipeline(sample_rate)
    model, labels = get_model(DEVICE)

    if sample_len > len(waveform):
        raise gr.Error("You must record for at least 6 seconds")
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = waveform.transpose(1,0)
        waveform = waveform.mean(axis=0, keepdims=True)
    else:
        waveform = np.expand_dims(waveform, 0)
    waveform = waveform[: ,:sample_len]
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min()) * 2 - 1
    waveform = waveform.astype("float32")
    waveform = torch.from_numpy(waveform)
    spectrogram = audio_pipeline(waveform)
    spectrogram = spectrogram.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        results = model(spectrogram)
    dance_mapping = get_dance_map()
    results = results.squeeze(0).detach().cpu().numpy()
    result_mask = results > threshold
    probs = results[result_mask]
    dances = labels[result_mask]
    
    return {dance_mapping[dance_id]:float(prob) for dance_id, prob in zip(dances, probs)} if len(dances) else "Couldn't find a dance."


def demo():
    title = "Dance Classifier"
    description = "Record 6 seconds of a song and find out what dance fits the music."
    with gr.Blocks() as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        with gr.Tab("Record Song"):
            mic_audio = gr.Audio(source="microphone", label="Song Recording")
            mic_submit = gr.Button("Predict")
            
        with gr.Tab("Upload Song") as t:
            audio_file = gr.Audio(label="Song Audio File")
            audio_file_submit = gr.Button("Predict")
        song_samples = Path(os.path.dirname(__file__), "assets", "song-samples")
        example_audio = [str(song) for song in song_samples.iterdir() if song.name[0] != '.']
        
        labels = gr.Label(label="Dances")

        gr.Markdown("## Examples")
        gr.Examples(
            examples=example_audio,
            inputs=audio_file,
            outputs=labels,
            fn=predict,
            )

        audio_file_submit.click(fn=predict, inputs=audio_file, outputs=labels)
        mic_submit.click(fn=predict, inputs=mic_audio, outputs=labels)

    return app


if __name__ == "__main__":
    demo().launch()
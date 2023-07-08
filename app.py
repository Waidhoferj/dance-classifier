from pathlib import Path
import gradio as gr
import numpy as np
import os
import pandas as pd
from functools import cache
from pathlib import Path
from models.residual import ResidualDancer
from models.training_environment import TrainingEnvironment
from preprocessing.pipelines import SpectrogramProductionPipeline, WaveformPreprocessing
import torch
from torch import nn
import yaml
import torchaudio

CONFIG_FILE = Path("models/weights/ResidualDancer/multilabel/config.yaml")

DANCE_MAPPING_FILE = Path("data/dance_mapping.csv")

MIN_DURATION = 3.0


class DancePredictor:
    def __init__(
        self,
        weight_path: str,
        labels: list[str],
        expected_duration=6,
        threshold=0.1,
        resample_frequency=16000,
        device="cpu",
    ):
        super().__init__()

        self.expected_duration = expected_duration
        self.threshold = threshold
        self.resample_frequency = resample_frequency

        self.labels = np.array(labels)
        self.device = device
        self.model = self.get_model(weight_path)
        self.process_waveform = WaveformPreprocessing(
            resample_frequency * expected_duration
        )
        self.extractor = SpectrogramProductionPipeline()

    def get_model(self, weight_path: str) -> nn.Module:
        weights = torch.load(weight_path, map_location=self.device)["state_dict"]
        n_classes = len(self.labels)
        # NOTE: Channels are not taken into account
        model = ResidualDancer(n_classes=n_classes).to(self.device)
        for key in list(weights):
            weights[
                key.replace(
                    "model.",
                    "",
                )
            ] = weights.pop(key)
        model.load_state_dict(weights, strict=False)
        return model.to(self.device).eval()

    @classmethod
    def from_config(cls, config_path: str) -> "DancePredictor":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        weight_path = config["checkpoint"]
        labels = sorted(config["dance_ids"])
        dance_mapping = get_dance_mapping(DANCE_MAPPING_FILE)
        labels = [dance_mapping[label] for label in labels]
        expected_duration = config.get("expected_duration", 6)
        threshold = config.get("threshold", 0.1)
        resample_frequency = config.get("resample_frequency", 16000)
        device = config.get("device", "cpu")
        return DancePredictor(
            weight_path,
            labels,
            expected_duration,
            threshold,
            resample_frequency,
            device,
        )

    @torch.no_grad()
    def __call__(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform]).T
        waveform = torch.from_numpy(waveform.T)
        waveform = torchaudio.functional.apply_codec(
            waveform, sample_rate, "wav", channels_first=True
        )

        waveform = torchaudio.functional.resample(
            waveform, sample_rate, self.resample_frequency
        )
        window_size = self.resample_frequency * self.expected_duration
        n_preds = int(waveform.shape[1] // (window_size / 2))
        step_size = int(waveform.shape[1] / n_preds)

        inputs = [
            waveform[:, i * step_size : i * step_size + window_size]
            for i in range(n_preds)
        ]
        features = [self.extractor(window) for window in inputs]
        features = torch.stack(features).to(self.device)
        results = self.model(features)
        # Convert to probabilities
        results = nn.functional.softmax(results, dim=1)
        # Take average prediction over all of the windows
        results = results.mean(dim=0)
        results = results.detach().cpu().numpy()

        result_mask = results > self.threshold
        probs = results[result_mask]
        dances = self.labels[result_mask]

        return {dance: float(prob) for dance, prob in zip(dances, probs)}


@cache
def get_model(config_path: str) -> DancePredictor:
    model = DancePredictor.from_config(config_path)
    return model


@cache
def get_dance_mapping(mapping_file: str) -> dict[str, str]:
    mapping_df = pd.read_csv(mapping_file)
    return {row["id"]: row["name"] for _, row in mapping_df.iterrows()}


def predict(audio: tuple[int, np.ndarray]) -> list[str]:
    if audio is None:
        return "Dance Not Found"
    sample_rate, waveform = audio
    duration = len(waveform) / sample_rate
    if duration < MIN_DURATION:
        return f"Please record at least {MIN_DURATION} seconds of audio"

    model = get_model(CONFIG_FILE)
    results = model(waveform, sample_rate)
    return results if len(results) else "Dance Not Found"


def demo():
    title = "Dance Classifier"
    description = "What should I dance to this song? Pass some audio to the Dance Classifier find out!"
    song_samples = Path(os.path.dirname(__file__), "assets", "song-samples")
    example_audio = [
        str(song) for song in song_samples.iterdir() if not song.name.startswith(".")
    ]
    all_dances = get_model(CONFIG_FILE).labels

    recording_interface = gr.Interface(
        fn=predict,
        inputs=gr.Audio(source="microphone", label="Song Recording"),
        outputs=gr.Label(label="Dances"),
        examples=example_audio,
    )
    uploading_interface = gr.Interface(
        fn=predict,
        inputs=gr.Audio(label="Song Audio File"),
        outputs=gr.Label(label="Dances"),
        examples=example_audio,
    )

    with gr.Blocks() as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        gr.TabbedInterface(
            [uploading_interface, recording_interface], ["Upload Song", "Record Song"]
        )
        with gr.Accordion("See all dances", open=False):
            gr.Markdown("\n".join(f"- {dance}" for dance in all_dances))

    return app


if __name__ == "__main__":
    demo().launch()

---
title: Dance Classifier
emoji: 💃
colorFrom: blue
colorTo: yellow
sdk: gradio
python_version: 3.10.8
sdk_version: 3.15.0
app_file: app.py
pinned: false
---

# Dance Classifier

Classifies the dance style that best accompanies a provided song. Users record or upload an audio clip and the model provides a list of matching dance styles.

## Getting Started

1. Clone this repo: `git clone https://github.com/Waidhoferj/dance-classifier`
2. Download git LSF files: `git lfs pull`
3. Download dependencies: `conda env create --file environment.yml`
4. Open environment: `conda activate dancer-classifier`
5. Start the demo application: `python app.py`

## Training

You can update and train models with the `train.py` script. The specific logic for training each model can be found in training functions located in the [models folder](./models/). You can customize and parameterize these training loops by directing the training script towards a custom [yaml config file](./models/config/).

```bash
# Train a model using a custom configuration
python train.py --config models/config/train_local.yaml
```

The training loops output the weights into either the `models/weights` or `lightning_logs` directories depending on the training script. You can then reference these pretrained weights for inference.

### Model Configuration

The YAML configuration files for training are located in [`models/config`](./models/config/). They specify the training environment, data, architecture, and hyperparameters of the model.

## Testing

See tests in the `tests` folder. Use Pytest to run the tests.

```bash
pytest
```

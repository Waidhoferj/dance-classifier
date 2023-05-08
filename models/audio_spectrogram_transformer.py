from transformers import ASTModel, AutoFeatureExtractor, ASTConfig, AutoModelForAudioClassification, TrainingArguments, Trainer
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")


class MultiModalAST(nn.Module):


    def __init__(self, labels, sample_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        id2label, label2id = get_id_label_mapping(labels)
        model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.ast_feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

        self.ast_model = ASTModel.from_pretrained(
        model_checkpoint, 
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
        )
        self.sample_rate = sample_rate
        
        self.bpm_model = nn.Sequential(
            nn.Linear(len(labels), 100),
            nn.Linear(100, 50)
        )

        out_dim = 50 # TODO: Calculate output dimension
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 100),
            nn.Linear(100, len(labels))
        )
    
    def vectorize_bpm(self, waveform):
        pass
    

    def forward(self, audio):

        bpm_vector = self.vectorize_bpm(audio)
        bpm_out = self.bpm_model(bpm_vector)

        spectrogram = self.ast_feature_extractor(audio)
        ast_out = self.ast_model(spectrogram)

        # Late fusion
        z = torch.cat([ast_out, bpm_out]) # Which dimension?
        return self.classifier(z)


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def get_id_label_mapping(labels:list[str]) -> tuple[dict, dict]:
    id2label = {str(i) : label for i, label in enumerate(labels)}
    label2id = {label : str(i) for i, label in enumerate(labels)}

    return id2label, label2id

def train(
        labels,
        train_ds, 
        test_ds, 
        output_dir="models/weights/ast",
        device="cpu",
        batch_size=128,
        epochs=10):
    id2label, label2id = get_id_label_mapping(labels)
    model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    preprocess_waveform = lambda wf : feature_extractor(wf, sampling_rate=train_ds.resample_frequency, padding="max_length", return_tensors="pt")
    train_ds.map(preprocess_waveform)
    test_ds.map(preprocess_waveform)

    model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint, 
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
).to(device)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=5,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        use_mps_device=device == "mps"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return model


    


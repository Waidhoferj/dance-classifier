from transformers import ASTFeatureExtractor, AutoFeatureExtractor, ASTConfig, AutoModelForAudioClassification, TrainingArguments, Trainer
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

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


    


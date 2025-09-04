#!/usr/bin/env python3
"""
Advanced Whisper Fine-tuning Script
Supports custom datasets, multiple languages, and optimized training
"""

import os
import torch
import torchaudio
from datasets import Dataset, DatasetDict, load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import evaluate
import numpy as np
from pathlib import Path
import json
import logging
from huggingface_hub import HfApi, create_repo
import wandb
from accelerate import Accelerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WhisperFineTuningConfig:
    """Configuration for Whisper fine-tuning"""

    # Model settings
    model_name: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"

    # Dataset settings
    dataset_name: Optional[str] = None  # e.g., "mozilla-foundation/common_voice_11_0"
    dataset_config: Optional[str] = None  # e.g., "en"
    custom_data_path: Optional[str] = None  # Path to custom dataset
    audio_column: str = "audio"
    text_column: str = "sentence"

    # Training settings
    output_dir: str = "./whisper-finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    logging_steps: int = 25
    eval_steps: int = 1000
    save_steps: int = 1000
    max_steps: int = -1

    # Data processing
    max_input_length: int = 30  # seconds
    max_target_length: int = 128  # tokens

    # Hardware optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

    # Hub settings
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "whisper-finetuning"


class WhisperFineTuner:
    def __init__(self, config: WhisperFineTuningConfig):
        self.config = config
        self.accelerator = Accelerator()

        # Initialize tokenizer and feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            config.model_name,
            language=config.language,
            task=config.task
        )
        self.processor = WhisperProcessor.from_pretrained(
            config.model_name,
            language=config.language,
            task=config.task
        )

        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        # Initialize metrics
        self.wer_metric = evaluate.load("wer")

        logger.info(f"Initialized WhisperFineTuner with model: {config.model_name}")

    def load_dataset(self) -> DatasetDict:
        """Load and prepare dataset for training"""

        if self.config.custom_data_path:
            # Load custom dataset
            dataset = self.load_custom_dataset()
        elif self.config.dataset_name:
            # Load from HuggingFace hub
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=["train", "validation"]
            )
            dataset = DatasetDict({
                "train": dataset[0],
                "validation": dataset[1]
            })
        else:
            raise ValueError("Either custom_data_path or dataset_name must be provided")

        # Cast audio column to Audio feature
        dataset = dataset.cast_column(self.config.audio_column, Audio(sampling_rate=16000))

        # Apply preprocessing
        dataset = dataset.map(
            self.prepare_dataset,
            remove_columns=dataset["train"].column_names,
            num_proc=4,
            desc="Preprocessing dataset"
        )

        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation")
        return dataset

    def load_custom_dataset(self) -> DatasetDict:
        """Load custom dataset from local files"""
        data_path = Path(self.config.custom_data_path)

        # Expected structure:
        # custom_data_path/
        # ├── train/
        # │   ├── audio/
        # │   └── transcripts.json
        # └── validation/
        #     ├── audio/
        #     └── transcripts.json

        def load_split(split_path: Path) -> Dataset:
            audio_dir = split_path / "audio"
            transcripts_file = split_path / "transcripts.json"

            if not transcripts_file.exists():
                raise FileNotFoundError(f"Transcripts file not found: {transcripts_file}")

            with open(transcripts_file, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)

            data = []
            for item in transcripts:
                audio_file = audio_dir / item["audio_file"]
                if audio_file.exists():
                    data.append({
                        "audio": str(audio_file),
                        "sentence": item["transcription"]
                    })

            return Dataset.from_list(data)

        train_dataset = load_split(data_path / "train")
        val_dataset = load_split(data_path / "validation")

        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })

    def prepare_dataset(self, batch):
        """Prepare dataset batch for training"""

        # Load and process audio
        audio = batch[self.config.audio_column]

        # Compute input features
        input_features = self.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]

        # Tokenize transcription
        labels = self.tokenizer(
            batch[self.config.text_column],
            max_length=self.config.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels
        }

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """Data collator for speech-to-text training"""
        processor: WhisperProcessor
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Split inputs and labels
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding token id's of the labels by -100
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # If bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    def compute_metrics(self, pred):
        """Compute WER metric"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def train(self):
        """Train the Whisper model"""

        # Initialize wandb if enabled
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project)

        # Load dataset
        dataset = self.load_dataset()

        # Data collator
        data_collator = self.DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            num_train_epochs=self.config.num_train_epochs,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to=["wandb"] if self.config.use_wandb else [],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_model_id,
            hub_token=self.config.hub_token,
            dataloader_num_workers=self.config.dataloader_num_workers,
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)

        # Push to hub if requested
        if self.config.push_to_hub and self.config.hub_model_id:
            logger.info(f"Pushing model to hub: {self.config.hub_model_id}")
            trainer.push_to_hub(commit_message="Fine-tuned Whisper model")

        logger.info("Training completed!")

        return trainer


class ProgressCallback(TrainerCallback):
    """Custom callback for training progress"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")


def main():
    """Main training function"""

    # Example configuration - customize as needed
    config = WhisperFineTuningConfig(
        model_name="openai/whisper-small",
        language="en",
        task="transcribe",

        # Use Common Voice dataset
        dataset_name="mozilla-foundation/common_voice_11_0",
        dataset_config="en",

        # Or use custom dataset
        # custom_data_path="./my_dataset",

        # Training settings
        output_dir="./whisper-finetuned-cv11",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=1e-5,

        # Hub settings
        push_to_hub=True,
        hub_model_id="your-username/whisper-finetuned-model",
        # hub_token="your_hf_token_here",  # Or set HF_TOKEN env var

        # Monitoring
        use_wandb=True,
        wandb_project="whisper-finetuning"
    )

    # Initialize fine-tuner
    fine_tuner = WhisperFineTuner(config)

    # Start training
    trainer = fine_tuner.train()

    return trainer


if __name__ == "__main__":
    main()
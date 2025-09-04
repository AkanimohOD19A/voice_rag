#!/usr/bin/env python3
"""
Script to push fine-tuned models to HuggingFace Hub
Includes model validation, testing, and comprehensive documentation
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import tempfile
import shutil

# HuggingFace
from huggingface_hub import HfApi, create_repo, upload_folder, login
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    pipeline
)
import librosa
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPusher:
    """Push fine-tuned models to HuggingFace Hub with validation"""

    def __init__(self,
                 model_path: str,
                 hub_model_id: str,
                 hf_token: Optional[str] = None,
                 private: bool = False):
        """
        Initialize ModelPusher

        Args:
            model_path: Path to the fine-tuned model directory
            hub_model_id: HuggingFace model ID (e.g., "username/model-name")
            hf_token: HuggingFace token (optional if already logged in)
            private: Whether to make the model private
        """
        self.model_path = Path(model_path)
        self.hub_model_id = hub_model_id
        self.private = private

        # Initialize HuggingFace API
        if hf_token:
            login(token=hf_token)

        self.api = HfApi()

        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        logger.info(f"Initialized ModelPusher for {hub_model_id}")

    def validate_model(self) -> Dict[str, any]:
        """Validate the model before pushing"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "model_info": {}
        }

        try:
            # Check required files
            required_files = [
                "config.json",
                "pytorch_model.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocab.json"
            ]

            missing_files = []
            for file in required_files:
                if not (self.model_path / file).exists():
                    missing_files.append(file)

            if missing_files:
                validation_results["errors"].append(f"Missing required files: {missing_files}")
                validation_results["valid"] = False

            # Load and validate model
            try:
                model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
                processor = WhisperProcessor.from_pretrained(self.model_path)

                # Get model info
                validation_results["model_info"] = {
                    "model_type": model.config.model_type,
                    "vocab_size": model.config.vocab_size,
                    "d_model": model.config.d_model,
                    "num_layers": model.config.encoder_layers,
                    "num_attention_heads": model.config.encoder_attention_heads,
                    "max_position_embeddings": model.config.max_position_embeddings,
                    "language": getattr(processor.tokenizer, "language", "unknown"),
                    "task": getattr(processor.tokenizer, "task", "unknown")
                }

                logger.info("Model validation passed")

            except Exception as e:
                validation_results["errors"].append(f"Failed to load model: {e}")
                validation_results["valid"] = False

            # Check model size
            model_size_mb = sum(f.stat().st_size for f in self.model_path.rglob('*') if f.is_file()) / (1024 * 1024)
            validation_results["model_info"]["size_mb"] = round(model_size_mb, 2)

            if model_size_mb > 1000:  # 1GB warning
                validation_results["warnings"].append(f"Large model size: {model_size_mb:.2f}MB")

        except Exception as e:
            validation_results["errors"].append(f"Validation error: {e}")
            validation_results["valid"] = False

        return validation_results

    def test_model_inference(self, test_audio_path: Optional[str] = None) -> Dict[str, any]:
        """Test model inference with sample audio"""
        test_results = {
            "success": False,
            "transcription": "",
            "inference_time": 0,
            "error": None
        }

        try:
            # Create test pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=str(self.model_path),
                tokenizer=str(self.model_path),
                feature_extractor=str(self.model_path),
                torch_dtype=torch.float32,
                device=-1  # CPU for testing
            )

            # Use test audio or generate synthetic
            if test_audio_path and os.path.exists(test_audio_path):
                audio_input = test_audio_path
            else:
                # Generate simple synthetic audio for testing
                audio_input = self._generate_test_audio()

            # Run inference
            start_time = datetime.now()
            result = pipe(audio_input)
            inference_time = (datetime.now() - start_time).total_seconds()

            test_results.update({
                "success": True,
                "transcription": result["text"],
                "inference_time": round(inference_time, 2)
            })

            logger.info(f"Model inference test passed: '{result['text']}'")

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Model inference test failed: {e}")

        return test_results

    def _generate_test_audio(self) -> str:
        """Generate synthetic test audio"""
        # Create simple sine wave audio
        sample_rate = 16000
        duration = 2  # seconds
        frequency = 440  # Hz

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        librosa.output.write_wav(temp_file.name, audio, sample_rate)

        return temp_file.name

    def create_model_card(self, validation_results: Dict, test_results: Dict) -> str:
        """Create comprehensive model card"""

        model_info = validation_results.get("model_info", {})

        model_card = f"""---
language: {model_info.get('language', 'en')}
license: apache-2.0
tags:
- whisper
- speech-recognition
- audio
- automatic-speech-recognition
- {model_info.get('language', 'en')}
datasets:
- custom-dataset
model-index:
- name: {self.hub_model_id}
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    metrics:
    - name: Test Inference Time
      type: inference_time
      value: {test_results.get('inference_time', 'N/A')}s
---

# {self.hub_model_id.split('/')[-1]}

This is a fine-tuned Whisper model for automatic speech recognition.

## Model Details

- **Model Type**: {model_info.get('model_type', 'whisper')}
- **Language**: {model_info.get('language', 'English')}
- **Task**: {model_info.get('task', 'transcribe')}
- **Base Model**: Whisper
- **Model Size**: {model_info.get('size_mb', 'Unknown')}MB
- **Vocabulary Size**: {model_info.get('vocab_size', 'Unknown')}
- **Architecture Details**:
  - Encoder Layers: {model_info.get('num_layers', 'Unknown')}
  - Attention Heads: {model_info.get('num_attention_heads', 'Unknown')}
  - Hidden Size: {model_info.get('d_model', 'Unknown')}

## Training Details

This model was fine-tuned using the Whisper fine-tuning script with the following characteristics:
- Fine-tuned for improved performance on specific audio domains
- Optimized for {model_info.get('language', 'English')} speech recognition
- Trained with gradient checkpointing for memory efficiency

## Usage

```python
from transformers import pipeline

# Load the model
pipe = pipeline(
    "automatic-speech-recognition", 
    model="{self.hub_model_id}",
    torch_dtype=torch.float16,
    device=0  # Use GPU if available
)

# Transcribe audio
result = pipe("path/to/audio/file.wav")
print(result["text"])
```

### With Transformers Library

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained("{self.hub_model_id}")
processor = WhisperProcessor.from_pretrained("{self.hub_model_id}")

# Process audio
audio_input = "path/to/audio/file.wav"
input_features = processor(audio_input, return_tensors="pt").input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])
```

## Performance

- **Test Inference Time**: {test_results.get('inference_time', 'N/A')}s (CPU)
- **Sample Transcription**: "{test_results.get('transcription', 'N/A')}"

## Technical Specifications

### Model Architecture
- **Encoder**: Transformer encoder with {model_info.get('num_layers', 'Unknown')} layers
- **Decoder**: Transformer decoder with causal attention
- **Attention Heads**: {model_info.get('num_attention_heads', 'Unknown')} per layer
- **Hidden Dimension**: {model_info.get('d_model', 'Unknown')}
- **Max Position Embeddings**: {model_info.get('max_position_embeddings', 'Unknown')}

### Input Requirements
- **Sampling Rate**: 16kHz
- **Audio Format**: WAV, MP3, FLAC, etc.
- **Max Audio Length**: 30 seconds per chunk
- **Input Features**: 80-dimensional log-mel spectrogram

## Fine-tuning Details

This model was fine-tuned using:
- Custom training pipeline with gradient checkpointing
- Mixed precision training (FP16)
- Optimized for {model_info.get('language', 'English')} language
- Task-specific optimization for {model_info.get('task', 'transcription')}

## Limitations and Bias

- Optimized for {model_info.get('language', 'English')} language
- Performance may vary on different audio qualities
- May have reduced performance on accents not well represented in training data
- Designed for {model_info.get('task', 'transcription')} task specifically

## Citation

If you use this model, please cite:

```bibtex
@misc{{{self.hub_model_id.replace('/', '_')},
  title={{{self.hub_model_id.split('/')[-1]}}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{self.hub_model_id}}}}}
}}
```

## Model Card Authors

This model card was generated automatically during the model upload process.

---

**Generated on**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Validation Status**: {'✅ Passed' if validation_results.get('valid') else '❌ Failed'}
**Test Status**: {'✅ Passed' if test_results.get('success') else '❌ Failed'}
"""

        return model_card

    def prepare_repository_files(self, validation_results: Dict, test_results: Dict) -> str:
        """Prepare all files for repository upload"""

        # Create temporary directory for upload
        temp_dir = tempfile.mkdtemp()
        upload_dir = Path(temp_dir) / "upload"
        upload_dir.mkdir()

        try:
            # Copy model files
            for item in self.model_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, upload_dir)
                elif item.is_dir():
                    shutil.copytree(item, upload_dir / item.name)

            # Create model card
            model_card = self.create_model_card(validation_results, test_results)
            with open(upload_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(model_card)

            # Create configuration files
            self._create_config_files(upload_dir, validation_results, test_results)

            logger.info(f"Prepared repository files in {upload_dir}")
            return str(upload_dir)

        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e

    def _create_config_files(self, upload_dir: Path, validation_results: Dict, test_results: Dict):
        """Create additional configuration files"""

        # Create .gitattributes for LFS
        gitattributes_content = """*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.mm filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
"""

        with open(upload_dir / ".gitattributes", "w") as f:
            f.write(gitattributes_content)

        # Create model metadata
        metadata = {
            "model_info": validation_results.get("model_info", {}),
            "validation_results": validation_results,
            "test_results": test_results,
            "upload_timestamp": datetime.now().isoformat(),
            "hub_model_id": self.hub_model_id
        }

        with open(upload_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def push_to_hub(self,
                    test_audio_path: Optional[str] = None,
                    commit_message: Optional[str] = None) -> Dict[str, any]:
        """Push model to HuggingFace Hub with full validation"""

        push_results = {
            "success": False,
            "model_url": f"https://huggingface.co/{self.hub_model_id}",
            "validation_results": None,
            "test_results": None,
            "error": None
        }

        try:
            # Step 1: Validate model
            logger.info("Step 1: Validating model...")
            validation_results = self.validate_model()
            push_results["validation_results"] = validation_results

            if not validation_results["valid"]:
                push_results["error"] = f"Model validation failed: {validation_results['errors']}"
                return push_results

            # Step 2: Test inference
            logger.info("Step 2: Testing model inference...")
            test_results = self.test_model_inference(test_audio_path)
            push_results["test_results"] = test_results

            # Step 3: Create repository
            logger.info("Step 3: Creating repository...")
            try:
                self.api.create_repo(
                    repo_id=self.hub_model_id,
                    private=self.private,
                    exist_ok=True
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise e

            # Step 4: Prepare files
            logger.info("Step 4: Preparing repository files...")
            upload_dir = self.prepare_repository_files(validation_results, test_results)

            # Step 5: Upload to hub
            logger.info("Step 5: Uploading to HuggingFace Hub...")

            if commit_message is None:
                commit_message = f"Upload fine-tuned Whisper model - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            self.api.upload_folder(
                folder_path=upload_dir,
                repo_id=self.hub_model_id,
                commit_message=commit_message,
                ignore_patterns=["*.pyc", "__pycache__/", ".git/"]
            )

            # Cleanup
            shutil.rmtree(Path(upload_dir).parent)

            push_results["success"] = True
            logger.info(f"Successfully pushed model to {push_results['model_url']}")

        except Exception as e:
            push_results["error"] = str(e)
            logger.error(f"Failed to push model: {e}")

        return push_results


def push_model_to_hub(model_path: str,
                      hub_model_id: str,
                      hf_token: Optional[str] = None,
                      private: bool = False,
                      test_audio_path: Optional[str] = None,
                      commit_message: Optional[str] = None) -> Dict[str, any]:
    """
    Convenience function to push a model to HuggingFace Hub

    Args:
        model_path: Path to the fine-tuned model
        hub_model_id: HuggingFace model ID (username/model-name)
        hf_token: HuggingFace token (optional)
        private: Whether to make the repository private
        test_audio_path: Path to test audio file (optional)
        commit_message: Custom commit message (optional)

    Returns:
        Dictionary with push results
    """

    pusher = ModelPusher(
        model_path=model_path,
        hub_model_id=hub_model_id,
        hf_token=hf_token,
        private=private
    )

    return pusher.push_to_hub(
        test_audio_path=test_audio_path,
        commit_message=commit_message
    )


class BatchModelPusher:
    """Push multiple models to HuggingFace Hub"""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        if hf_token:
            login(token=hf_token)

    def push_multiple_models(self, model_configs: List[Dict]) -> Dict[str, Dict]:
        """
        Push multiple models to hub

        Args:
            model_configs: List of model configurations, each containing:
                - model_path: Path to model
                - hub_model_id: HuggingFace model ID
                - private: Whether to make private (optional)
                - test_audio_path: Test audio path (optional)
                - commit_message: Commit message (optional)

        Returns:
            Dictionary mapping model_id to push results
        """
        results = {}

        for i, config in enumerate(model_configs, 1):
            logger.info(f"Processing model {i}/{len(model_configs)}: {config['hub_model_id']}")

            try:
                pusher = ModelPusher(
                    model_path=config['model_path'],
                    hub_model_id=config['hub_model_id'],
                    hf_token=self.hf_token,
                    private=config.get('private', False)
                )

                result = pusher.push_to_hub(
                    test_audio_path=config.get('test_audio_path'),
                    commit_message=config.get('commit_message')
                )

                results[config['hub_model_id']] = result

            except Exception as e:
                results[config['hub_model_id']] = {
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"Failed to push {config['hub_model_id']}: {e}")

        return results


# CLI interface
def main():
    """Command line interface for model pushing"""
    import argparse

    parser = argparse.ArgumentParser(description="Push fine-tuned Whisper model to HuggingFace Hub")
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--hub-model-id", required=True, help="HuggingFace model ID (username/model-name)")
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--test-audio", help="Path to test audio file")
    parser.add_argument("--commit-message", help="Custom commit message")
    parser.add_argument("--validate-only", action="store_true", help="Only validate model without pushing")

    args = parser.parse_args()

    # Get token from args or environment
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    if not hf_token and not args.validate_only:
        logger.warning("No HuggingFace token provided. Make sure you're logged in with `huggingface-cli login`")

    try:
        pusher = ModelPusher(
            model_path=args.model_path,
            hub_model_id=args.hub_model_id,
            hf_token=hf_token,
            private=args.private
        )

        if args.validate_only:
            # Only validate
            logger.info("Validating model...")
            validation_results = pusher.validate_model()
            test_results = pusher.test_model_inference(args.test_audio)

            print("\n" + "=" * 50)
            print("VALIDATION RESULTS")
            print("=" * 50)
            print(f"Valid: {validation_results['valid']}")
            if validation_results['errors']:
                print(f"Errors: {validation_results['errors']}")
            if validation_results['warnings']:
                print(f"Warnings: {validation_results['warnings']}")

            print(f"\nModel Info: {json.dumps(validation_results['model_info'], indent=2)}")

            print("\n" + "=" * 50)
            print("TEST RESULTS")
            print("=" * 50)
            print(f"Success: {test_results['success']}")
            print(f"Inference Time: {test_results['inference_time']}s")
            print(f"Transcription: '{test_results['transcription']}'")
            if test_results['error']:
                print(f"Error: {test_results['error']}")

        else:
            # Push to hub
            logger.info("Pushing model to HuggingFace Hub...")
            results = pusher.push_to_hub(
                test_audio_path=args.test_audio,
                commit_message=args.commit_message
            )

            print("\n" + "=" * 50)
            print("PUSH RESULTS")
            print("=" * 50)
            print(f"Success: {results['success']}")
            print(f"Model URL: {results['model_url']}")

            if results['error']:
                print(f"Error: {results['error']}")
                return 1

            print("\nValidation Results:")
            print(f"  Valid: {results['validation_results']['valid']}")
            print(f"  Model Size: {results['validation_results']['model_info'].get('size_mb', 'Unknown')}MB")

            print("\nTest Results:")
            print(f"  Inference Success: {results['test_results']['success']}")
            print(f"  Inference Time: {results['test_results']['inference_time']}s")
            print(f"  Sample Transcription: '{results['test_results']['transcription']}'")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


# Example usage functions
def example_single_model_push():
    """Example: Push a single model"""

    # Configuration
    model_path = "./whisper-finetuned"
    hub_model_id = "your-username/whisper-finetuned-model"
    hf_token = "your_hf_token_here"  # Or set HF_TOKEN environment variable

    # Push model
    results = push_model_to_hub(
        model_path=model_path,
        hub_model_id=hub_model_id,
        hf_token=hf_token,
        private=False,
        test_audio_path="./test_audio.wav",  # Optional
        commit_message="Initial upload of fine-tuned Whisper model"
    )

    if results["success"]:
        print(f"✅ Model successfully pushed to {results['model_url']}")
        print(f"Model size: {results['validation_results']['model_info']['size_mb']}MB")
        print(f"Test transcription: '{results['test_results']['transcription']}'")
    else:
        print(f"❌ Failed to push model: {results['error']}")


def example_batch_model_push():
    """Example: Push multiple models in batch"""

    batch_pusher = BatchModelPusher(hf_token="your_hf_token_here")

    model_configs = [
        {
            "model_path": "./whisper-english-finetuned",
            "hub_model_id": "your-username/whisper-english-v1",
            "private": False,
            "commit_message": "English fine-tuned Whisper model v1"
        },
        {
            "model_path": "./whisper-multilingual-finetuned",
            "hub_model_id": "your-username/whisper-multilingual-v1",
            "private": True,
            "test_audio_path": "./test_multilingual.wav",
            "commit_message": "Multilingual fine-tuned Whisper model v1"
        }
    ]

    results = batch_pusher.push_multiple_models(model_configs)

    for model_id, result in results.items():
        if result["success"]:
            print(f"✅ {model_id}: Successfully pushed")
        else:
            print(f"❌ {model_id}: Failed - {result['error']}")


if __name__ == "__main__":
    main()
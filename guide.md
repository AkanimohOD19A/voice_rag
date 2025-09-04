# 🚀 Voice RAG System - Complete Deployment Guide

This guide walks you through the complete refactoring and deployment process for your Voice RAG system.

## 📋 Overview

The refactored system includes:

1. **Fine-tuning Script** - Advanced Whisper model fine-tuning
2. **Enhanced RAG System** - Document upload and processing capabilities
3. **Model Push Script** - Automated HuggingFace Hub deployment
4. **Production Gradio App** - HuggingFace Space-ready interface

## 🏗️ Project Structure

```
voice-rag-system/
├── training/
│   ├── fine_tune_whisper.py       # Fine-tuning script
│   ├── train_config.yaml          # Training configuration
│   └── datasets/                  # Training data
├── rag_system/
│   ├── advanced_rag.py            # Enhanced RAG system
│   ├── document_processor.py      # Document processing utilities
│   └── vector_stores.py           # Vector store implementations
├── deployment/
│   ├── push_to_hub.py             # Model deployment script
│   ├── model_validation.py        # Model validation utilities
│   └── hub_configs/               # HuggingFace configurations
├── gradio_app/
│   ├── app.py                     # Main Gradio application
│   ├── interface.py               # UI components
│   └── utils.py                   # Helper functions
├── hf_space/
│   ├── app.py                     # HF Space entry point
│   ├── requirements.txt           # Dependencies
│   ├── README.md                  # Space documentation
│   ├── config.yaml                # Configuration
│   └── Dockerfile                 # Container setup
└── examples/
    ├── quick_start.py             # Quick start example
    ├── batch_processing.py        # Batch operations
    └── custom_integration.py      # Custom integration
```

## 🎯 Step-by-Step Deployment

### Step 1: Fine-tune Your Whisper Model

```bash
# 1. Prepare your dataset
# Structure: dataset/train/audio/ and dataset/train/transcripts.json
# Example transcripts.json:
# [
#   {"audio_file": "audio1.wav", "transcription": "Hello world"},
#   {"audio_file": "audio2.wav", "transcription": "How are you?"}
# ]

# 2. Run fine-tuning
python fine_tune_whisper.py \
    --model-name "openai/whisper-small" \
    --custom-data-path "./dataset" \
    --output-dir "./whisper-finetuned" \
    --epochs 3 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --push-to-hub \
    --hub-model-id "your-username/whisper-finetuned-model"
```

**Example Configuration:**

```python
from fine_tune_whisper import WhisperFineTuningConfig, WhisperFineTuner

# Create configuration
config = WhisperFineTuningConfig(
    model_name="openai/whisper-small",
    language="en",
    custom_data_path="./my_dataset",
    output_dir="./whisper-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    push_to_hub=True,
    hub_model_id="your-username/whisper-finetuned-model",
    use_wandb=True  # For experiment tracking
)

# Initialize and train
fine_tuner = WhisperFineTuner(config)
trainer = fine_tuner.train()
```

### Step 2: Set Up Enhanced RAG System

```python
from advanced_rag import AdvancedRAGSystem, RAGConfig

# Configure RAG system
rag_config = RAGConfig(
    embedding_model="all-MiniLM-L6-v2",
    vector_store_type="faiss",
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5,
    similarity_threshold=0.3
)

# Initialize system
rag_system = AdvancedRAGSystem(rag_config)

# Process documents
documents = ["doc1.pdf", "doc2.docx", "data.csv"]
results = rag_system.process_uploaded_files(documents)

print(f"Processed {results['total_chunks']} chunks from {len(results['processed_files'])} files")

# Query the system
query = "What are the main findings?"
result = rag_system.process_query_complete(query)
print(result['response'])
```

### Step 3: Push Model to HuggingFace Hub

```bash
# Validate and push your fine-tuned model
python push_to_hub.py \
    --model-path "./whisper-finetuned" \
    --hub-model-id "your-username/whisper-finetuned-model" \
    --test-audio "./test_audio.wav" \
    --commit-message "Fine-tuned Whisper for domain-specific ASR"

# Or just validate without pushing
python push_to_hub.py \
    --model-path "./whisper-finetuned" \
    --hub-model-id "your-username/whisper-finetuned-model" \
    --validate-only
```

**Programmatic Usage:**

```python
from push_to_hub import push_model_to_hub

# Push single model
results = push_model_to_hub(
    model_path="./whisper-finetuned",
    hub_model_id="your-username/whisper-finetuned-model",
    hf_token="your_hf_token",
    private=False,
    test_audio_path="./test.wav"
)

if results["success"]:
    print(f"✅ Model pushed to {results['model_url']}")
else:
    print(f"❌ Error: {results['error']}")
```

### Step 4: Deploy to HuggingFace Spaces

#### 4.1 Create Space Repository

```bash
# Create new space on HuggingFace
huggingface-cli repo create your-username/voice-rag-system --type space --sdk gradio

# Clone the space repository
git clone https://huggingface.co/spaces/your-username/voice-rag-system
cd voice-rag-system
```

#### 4.2 Copy Application Files

```bash
# Copy all necessary files to the space directory
cp hf_space/app.py .
cp hf_space/requirements.txt .
cp hf_space/README.md .
cp hf_space/config.yaml .
cp gradio_app/* .
```

#### 4.3 Update Configuration

Edit `app.py` to use your fine-tuned model:

```python
# Update model ID in VoiceRAGSpace.__init__
model_id = "your-username/whisper-finetuned-model"
```

#### 4.4 Deploy

```bash
# Add files and deploy
git add .
git commit -m "Deploy Voice RAG System to Spaces"
git push
```

Your space will be available at: `https://huggingface.co/spaces/your-username/voice-rag-system`

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Set in your HuggingFace Space settings
HF_TOKEN=your_huggingface_token
TRANSFORMERS_CACHE=/tmp/cache
GRADIO_SERVER_PORT=7860
```

### Custom Model Integration

```python
# In your Gradio app, update model loading:
class VoiceRAGSpace:
    def setup_models(self):
        # Try your fine-tuned model first
        try:
            model_id = os.getenv("CUSTOM_MODEL_ID", "your-username/whisper-finetuned-model")
            self.speech_to_text = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch.float16,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            # Fallback to base model
            logger.warning(f"Failed to load custom model: {e}")
            self.speech_to_text = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base"
            )
```

### Performance Optimization

```python
# GPU optimization for Spaces with GPU
if torch.cuda.is_available():
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Use mixed precision
    from torch.cuda.amp import autocast
    
    # In your processing functions:
    with autocast():
        embeddings = self.embedder.encode(texts)
```

## 📊 Monitoring and Analytics

### Add Usage Tracking

```python
# In your Gradio app
import time
from datetime import datetime

def track_usage(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        processing_time = time.time() - start_time
        
        # Log usage metrics
        logger.info(f"Function: {func.__name__}, Time: {processing_time:.2f}s")
        return result
    return wrapper

@track_usage
def process_audio_query(audio_file):
    # Your processing logic
    pass
```

### Performance Dashboard

```python
# Add performance monitoring tab to Gradio
with gr.TabItem("📊 Analytics"):
    gr.Plot(value=get_usage_stats, every=1)  # Update every second
    gr.Textbox(value=get_system_metrics, every=5)  # Update every 5 seconds
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```python
   # Add error handling and fallback
   try:
       model = pipeline("automatic-speech-recognition", model=model_id)
   except Exception as e:
       logger.error(f"Model loading failed: {e}")
       model = pipeline("automatic-speech-recognition", model="openai/whisper-base")
   ```

2. **Out of Memory**
   ```python
   # Reduce batch size and enable gradient checkpointing
   config.per_device_train_batch_size = 8
   config.gradient_checkpointing = True
   config.fp16 = True
   ```

3. **Slow Processing**
   ```python
   # Enable GPU and optimize settings
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model.to(device)
   
   # Use smaller chunk sizes
   chunk_size = 512  # Instead of 1000
   ```

### Debug Mode

```python
# Enable debug mode in development
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug information to responses
def debug_response(query, response, retrieved_docs):
    debug_info = {
        "query": query,
        "response_length": len(response),
        "docs_retrieved": len(retrieved_docs),
        "model_device": str(model.device),
        "timestamp": datetime.now().isoformat()
    }
    logger.debug(f"Debug info: {debug_info}")
    return debug_info
```

## 🎉 Testing Your Deployment

### Local Testing

```python
# Test fine-tuned model
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="./whisper-finetuned")
result = pipe("test_audio.wav")
print(f"Transcription: {result['text']}")

# Test RAG system
from advanced_rag import create_rag_system

rag = create_rag_system()
rag.add_text_documents(["Test document content"])
result = rag.process_query_complete("What is this about?")
print(result['response'])
```

### Integration Testing

```bash
# Test complete pipeline
python test_integration.py \
    --model-path "./whisper-finetuned" \
    --test-documents "./test_docs/" \
    --test-audio "./test_audio.wav" \
    --test-queries "./test_queries.txt"
```

## 📈 Next Steps

1. **Scale Up**: Consider using more powerful embedding models
2. **Add Features**: Implement conversation memory, multi-language support
3. **Optimize**: Fine-tune chunk sizes and similarity thresholds
4. **Monitor**: Set up logging and analytics
5. **Iterate**: Collect user feedback and improve the model

## 💡 Tips for Success

- **Start Small**: Begin with a small dataset and basic configuration
- **Monitor Performance**: Track processing times and accuracy
- **User Feedback**: Collect feedback to improve the system
- **Documentation**: Keep your model cards and documentation updated
- **Version Control**: Use semantic versioning for your models
- **Backup**: Keep backups of your trained models and data

---

**🎯 Your Voice RAG system is now ready for production deployment!**

The refactored system provides:
- ✅ Professional fine-tuning capabilities
- ✅ Advanced document processing
- ✅ Automated model deployment
- ✅ Production-ready Gradio interface
- ✅ HuggingFace Spaces integration
- ✅ Comprehensive monitoring and debugging

Happy building! 🚀
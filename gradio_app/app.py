#!/usr/bin/env python3
"""
HuggingFace Space Gradio App for Voice RAG System
Production-ready interface with document upload and voice interaction
"""

import os
import gradio as gr
import torch
import numpy as np
from datetime import datetime
import tempfile
import json
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

# Core imports
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from sentence_transformers import SentenceTransformer
import faiss
import requests
from bs4 import BeautifulSoup

# Document processing
import PyPDF2
import docx
from PIL import Image
import pytesseract

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceRAGSpace:
    """Voice RAG system optimized for HuggingFace Spaces"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_models()
        self.documents = []
        self.document_embeddings = None
        self.faiss_index = None
        self.processed_files = set()

        logger.info(f"Initialized VoiceRAGSpace on {self.device}")

    def setup_models(self):
        """Setup models optimized for HuggingFace Spaces"""
        try:
            # Use your fine-tuned model or fallback to base model
            model_id = "your-username/whisper-finetuned-model"  # Replace with your model

            # Try to load your model first
            try:
                self.speech_to_text = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device=0 if self.device == "cuda" else -1,
                    chunk_length_s=30
                )
                logger.info(f"Loaded fine-tuned model: {model_id}")
            except:
                # Fallback to base Whisper model
                logger.info("Loading fallback Whisper model...")
                self.speech_to_text = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device=0 if self.device == "cuda" else -1
                )

            # Setup embedding model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise

    def process_uploaded_file(self, file_path: str) -> Dict:
        """Process a single uploaded file"""
        try:
            file_path = Path(file_path)

            # Check if already processed
            if file_path.name in self.processed_files:
                return {"status": "already_processed", "filename": file_path.name}

            # Extract text based on file type
            extension = file_path.suffix.lower()

            if extension == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif extension == '.docx':
                text = self._extract_docx_text(file_path)
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif extension in ['.jpg', '.jpeg', '.png']:
                text = self._extract_image_text(file_path)
            else:
                return {"status": "unsupported", "filename": file_path.name}

            if not text.strip():
                return {"status": "no_text", "filename": file_path.name}

            # Chunk text
            chunks = self._chunk_text(text, chunk_size=1000, overlap=200)

            # Generate embeddings
            embeddings = self.embedder.encode(chunks, convert_to_numpy=True)

            # Add to document store
            for chunk, embedding in zip(chunks, embeddings):
                self.documents.append({
                    "content": chunk,
                    "filename": file_path.name,
                    "file_type": extension
                })

            # Update embeddings and index
            if self.document_embeddings is None:
                self.document_embeddings = embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings])

            self._update_faiss_index()
            self.processed_files.add(file_path.name)

            return {
                "status": "success",
                "filename": file_path.name,
                "chunks": len(chunks),
                "text_length": len(text)
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {"status": "error", "filename": file_path.name, "error": str(e)}

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""

    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return ""

    def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple text chunking"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def _update_faiss_index(self):
        """Update FAISS index with new embeddings"""
        if self.document_embeddings is not None:
            dimension = self.document_embeddings.shape[1]

            # Create new index
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.GpuIndexFlatIP(res, dimension)
                except:
                    self.faiss_index = faiss.IndexFlatIP(dimension)
            else:
                self.faiss_index = faiss.IndexFlatIP(dimension)

            # Add embeddings
            self.faiss_index.add(self.document_embeddings.astype(np.float32))

    def transcribe_audio(self, audio_file) -> str:
        """Transcribe audio to text"""
        try:
            if audio_file is None:
                return "No audio file provided"

            result = self.speech_to_text(audio_file)
            return result["text"].strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Transcription error: {e}"

    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        if not self.documents or self.faiss_index is None:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)

            # Search
            scores, indices = self.faiss_index.search(
                query_embedding.astype(np.float32),
                min(k, len(self.documents))
            )

            results = []
            for idx, score in zip(indices[0], scores[0]):
                if score > 0.3:  # Similarity threshold
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(score)
                    results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def web_search(self, query: str, num_results: int = 3) -> List[Dict]:
        """Simple web search using DuckDuckGo API"""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=5)
            data = response.json()

            results = []

            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractSource', 'DuckDuckGo'),
                    'content': data['Abstract'][:300],
                    'url': data.get('AbstractURL', ''),
                    'source': 'web'
                })

            for topic in data.get('RelatedTopics', [])[:num_results - 1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('FirstURL', '').split('/')[-1] or 'Related',
                        'content': topic['Text'][:200],
                        'url': topic.get('FirstURL', ''),
                        'source': 'web'
                    })

            return results

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    def generate_response(self, query: str, retrieved_docs: List[Dict],
                          web_results: List[Dict]) -> str:
        """Generate response from retrieved information"""

        response_parts = [f"🎯 **Query:** {query}\n"]

        if retrieved_docs:
            response_parts.append("📚 **From Your Documents:**")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                filename = doc.get('filename', 'Unknown')
                score = doc.get('similarity_score', 0)
                content_preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                response_parts.append(f"{i}. **{filename}** (similarity: {score:.2f})")
                response_parts.append(f"   {content_preview}")

        if web_results:
            response_parts.append("\n🌐 **From Web Search:**")
            for i, result in enumerate(web_results[:2], 1):
                response_parts.append(f"{i}. **{result['title']}**")
                response_parts.append(f"   {result['content']}")
                if result['url']:
                    response_parts.append(f"   🔗 {result['url']}")

        if not retrieved_docs and not web_results:
            response_parts.append(
                "❌ No relevant information found. Try uploading relevant documents or rephrasing your question.")

        return "\n".join(response_parts)


# Global instance
voice_rag = VoiceRAGSpace()


def process_audio_query(audio_file):
    """Process voice query"""
    if audio_file is None:
        return "Please provide an audio file", "", "No audio provided"

    start_time = datetime.now()

    try:
        # Transcribe audio
        transcription = voice_rag.transcribe_audio(audio_file)

        if transcription.startswith("Transcription error"):
            return transcription, "", "Audio processing failed"

        # Retrieve documents
        retrieved_docs = voice_rag.retrieve_documents(transcription, k=5)

        # Web search
        web_results = voice_rag.web_search(transcription, num_results=3)

        # Generate response
        response = voice_rag.generate_response(transcription, retrieved_docs, web_results)

        # Performance info
        processing_time = (datetime.now() - start_time).total_seconds()
        perf_info = f"""⚡ **Performance:**
• Processing Time: {processing_time:.2f}s
• Retrieved Documents: {len(retrieved_docs)}
• Web Results: {len(web_results)}
• Total Documents in KB: {len(voice_rag.documents)}
"""

        return transcription, response, perf_info

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        return f"Error: {e}", "", f"Failed after {error_time:.2f}s"


def process_text_query(text_query):
    """Process text query"""
    if not text_query.strip():
        return "Please enter a question", "No query provided"

    start_time = datetime.now()

    try:
        # Retrieve documents
        retrieved_docs = voice_rag.retrieve_documents(text_query, k=5)

        # Web search
        web_results = voice_rag.web_search(text_query, num_results=3)

        # Generate response
        response = voice_rag.generate_response(text_query, retrieved_docs, web_results)

        # Performance info
        processing_time = (datetime.now() - start_time).total_seconds()
        perf_info = f"""⚡ **Performance:**
• Processing Time: {processing_time:.2f}s
• Retrieved Documents: {len(retrieved_docs)}
• Web Results: {len(web_results)}
• Total Documents in KB: {len(voice_rag.documents)}
"""

        return response, perf_info

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        return f"Error: {e}", f"Failed after {error_time:.2f}s"


def upload_documents(files):
    """Handle document upload"""
    if not files:
        return "No files uploaded", ""

    results = {
        "processed": [],
        "failed": [],
        "total_chunks": 0
    }

    for file in files:
        result = voice_rag.process_uploaded_file(file.name)

        if result["status"] == "success":
            results["processed"].append({
                "filename": result["filename"],
                "chunks": result["chunks"],
                "size": result["text_length"]
            })
            results["total_chunks"] += result["chunks"]
        else:
            results["failed"].append({
                "filename": result["filename"],
                "reason": result.get("error", result["status"])
            })

    # Create summary
    summary_parts = [
        f"📁 **Upload Summary:**",
        f"• Successfully processed: {len(results['processed'])} files",
        f"• Total chunks added: {results['total_chunks']}",
        f"• Failed: {len(results['failed'])} files"
    ]

    if results["processed"]:
        summary_parts.append("\n✅ **Processed Files:**")
        for file_info in results["processed"]:
            summary_parts.append(f"• {file_info['filename']}: {file_info['chunks']} chunks ({file_info['size']} chars)")

    if results["failed"]:
        summary_parts.append("\n❌ **Failed Files:**")
        for file_info in results["failed"]:
            summary_parts.append(f"• {file_info['filename']}: {file_info['reason']}")

    # System stats
    stats = f"""📊 **System Status:**
• Total Documents: {len(voice_rag.documents)}
• Processed Files: {len(voice_rag.processed_files)}
• Device: {voice_rag.device}
• Model: Whisper + SentenceTransformer
"""

    return "\n".join(summary_parts), stats


def clear_documents():
    """Clear all documents"""
    voice_rag.documents.clear()
    voice_rag.document_embeddings = None
    voice_rag.faiss_index = None
    voice_rag.processed_files.clear()

    return "🗑️ All documents cleared from the system", "📊 **System Status:** Empty knowledge base"


# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
            title="🎤 Voice RAG System",
            theme=gr.themes.Soft(),
            css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🎤 Voice RAG System

        **AI-powered voice and text question-answering with document knowledge base**

        📁 Upload documents → 🎤 Ask questions via voice or text → 🤖 Get intelligent answers
        """, elem_classes=["header"])

        with gr.Tabs():
            # Document Upload Tab
            with gr.TabItem("📁 Document Upload", id=0):
                gr.Markdown("### Upload Documents to Knowledge Base")

                with gr.Row():
                    with gr.Column(scale=3):
                        file_upload = gr.File(
                            label="📎 Upload Documents",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"],
                            height=200
                        )

                        with gr.Row():
                            upload_btn = gr.Button("📤 Process Documents", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear All Documents", variant="secondary")

                    with gr.Column(scale=2):
                        upload_status = gr.Textbox(
                            label="📋 Upload Status",
                            lines=10,
                            interactive=False
                        )

                system_stats = gr.Textbox(
                    label="📊 System Statistics",
                    lines=6,
                    interactive=False
                )

                # Event handlers
                upload_btn.click(
                    upload_documents,
                    inputs=[file_upload],
                    outputs=[upload_status, system_stats]
                )

                clear_btn.click(
                    clear_documents,
                    outputs=[upload_status, system_stats]
                )

            # Voice Query Tab
            with gr.TabItem("🎤 Voice Query", id=1):
                gr.Markdown("### Ask Questions Using Your Voice")

                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="🎤 Record or Upload Audio",
                            sources=["microphone", "upload"],
                            type="filepath"
                        )

                        voice_btn = gr.Button("🎯 Process Voice Query", variant="primary")

                    with gr.Column():
                        transcription_output = gr.Textbox(
                            label="📝 Transcription",
                            lines=3,
                            interactive=False
                        )

                voice_response = gr.Textbox(
                    label="🤖 AI Response",
                    lines=12,
                    interactive=False
                )

                voice_performance = gr.Textbox(
                    label="⚡ Performance Metrics",
                    lines=5,
                    interactive=False
                )

                # Event handler
                voice_btn.click(
                    process_audio_query,
                    inputs=[audio_input],
                    outputs=[transcription_output, voice_response, voice_performance]
                )

            # Text Query Tab
            with gr.TabItem("💬 Text Query", id=2):
                gr.Markdown("### Ask Questions Using Text")

                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="❓ Enter your question",
                            placeholder="What would you like to know?",
                            lines=3
                        )

                        text_btn = gr.Button("🎯 Process Text Query", variant="primary")

                    with gr.Column():
                        text_performance = gr.Textbox(
                            label="⚡ Performance Metrics",
                            lines=5,
                            interactive=False
                        )

                text_response = gr.Textbox(
                    label="🤖 AI Response",
                    lines=12,
                    interactive=False
                )

                # Event handlers
                text_btn.click(
                    process_text_query,
                    inputs=[text_input],
                    outputs=[text_response, text_performance]
                )

                # Enter key support
                text_input.submit(
                    process_text_query,
                    inputs=[text_input],
                    outputs=[text_response, text_performance]
                )

            # System Info Tab
            with gr.TabItem("ℹ️ System Info", id=3):
                gr.Markdown("### System Information and Usage Guide")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        #### 🚀 **Features**
                        - **Voice Recognition**: Upload audio or record directly
                        - **Document Processing**: PDF, DOCX, TXT, Images (OCR)
                        - **Semantic Search**: Find relevant information from your documents
                        - **Web Search**: Real-time web results for current information
                        - **Multi-modal**: Both voice and text input supported

                        #### 📋 **How to Use**
                        1. **Upload Documents**: Go to Document Upload tab and add your files
                        2. **Ask Questions**: Use Voice Query or Text Query tabs
                        3. **Get Answers**: System combines your documents with web search

                        #### 🔧 **Supported Formats**
                        - **Text**: PDF, DOCX, TXT
                        - **Images**: JPG, PNG (with OCR)
                        - **Audio**: WAV, MP3, M4A

                        #### 🎯 **Tips for Best Results**
                        - Upload relevant documents before asking questions
                        - Speak clearly for voice queries
                        - Ask specific, focused questions
                        - Check document upload status before querying
                        """)

                    with gr.Column():
                        gr.Markdown("""
                        #### ⚙️ **Technical Details**
                        - **Speech Recognition**: Fine-tuned Whisper model
                        - **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
                        - **Vector Store**: FAISS for fast similarity search
                        - **Web Search**: DuckDuckGo API integration
                        - **Processing**: GPU-accelerated when available

                        #### 🔒 **Privacy & Security**
                        - Documents are processed locally in the session
                        - No permanent storage of uploaded files
                        - Audio is processed temporarily
                        - Session data is cleared when you leave

                        #### 🐛 **Troubleshooting**
                        - **No results**: Make sure documents are uploaded first
                        - **Audio issues**: Check microphone permissions
                        - **Large files**: Split large documents into smaller parts
                        - **Processing slow**: Reduce number of documents or file sizes

                        #### 📞 **Support**
                        If you encounter issues:
                        1. Check system statistics in Document Upload tab
                        2. Try clearing documents and re-uploading
                        3. Use text query if voice recognition fails
                        4. Refresh page for a clean restart
                        """)

                # Real-time system stats
                def get_current_stats():
                    return f"""
                    **Current System Status:**
                    - Documents in Knowledge Base: {len(voice_rag.documents)}
                    - Processed Files: {len(voice_rag.processed_files)}
                    - Computing Device: {voice_rag.device}
                    - Embedding Model: all-MiniLM-L6-v2
                    - Vector Store: {"FAISS (Ready)" if voice_rag.faiss_index else "FAISS (Empty)"}
                    - Last Updated: {datetime.now().strftime("%H:%M:%S")}
                    """

                current_stats = gr.Textbox(
                    label="📊 Current System Status",
                    value=get_current_stats(),
                    lines=8,
                    interactive=False
                )

                refresh_stats_btn = gr.Button("🔄 Refresh Stats")
                refresh_stats_btn.click(
                    lambda: get_current_stats(),
                    outputs=[current_stats]
                )

        # Footer
        gr.Markdown("""
        ---
        **Voice RAG System** - Powered by 🤗 Transformers, Whisper, and SentenceTransformers  
        *Upload documents, ask questions via voice or text, get intelligent AI-powered answers*
        """)

    return demo


# Launch configuration for HuggingFace Spaces
def launch_space():
    """Launch the Gradio app for HuggingFace Spaces"""

    # Create interface
    demo = create_interface()

    # Launch with HuggingFace Spaces optimized settings
    demo.launch(
        server_name="0.0.0.0",  # Required for HF Spaces
        server_port=7860,  # Default HF Spaces port
        share=False,  # No need to share on Spaces
        debug=False,  # Disable debug in production
        show_error=True,  # Show errors to users
        quiet=False,  # Show startup logs
        favicon_path=None,  # Default favicon
        auth=None,  # No authentication by default
        max_threads=40,  # Handle concurrent users
        show_tips=False,  # Clean interface
        enable_queue=True,  # Enable queuing for better performance
        max_size=20  # Max queue size
    )


if __name__ == "__main__":
    launch_space()
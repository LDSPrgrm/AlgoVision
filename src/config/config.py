import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OUTPUT_DIR = "output"
    CONTEXT_LEARNING_PATH = "data/context_learning"
    CHROMA_DB_PATH = "data/rag/chroma_db"
    MANIM_DOCS_PATH = "data/rag/manim_docs"
    EMBEDDING_MODEL = "azure/text-embedding-3-large"
    
    # Kokoro TTS configurations
    KOKORO_MODEL_PATH = "src/tts/kokoro-v1.0.onnx"
    KOKORO_VOICES_PATH = "src/tts/voices-v1.0.bin"
    KOKORO_DEFAULT_VOICE = "af_heart"
    KOKORO_DEFAULT_SPEED = float("1.0")
    KOKORO_DEFAULT_LANG = "en-us"
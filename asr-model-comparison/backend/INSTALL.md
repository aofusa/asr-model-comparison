# Backend Installation Guide (Real ASR Models)

This guide explains how to install the **actual** ASR model dependencies after the basic TDD environment is set up.

**Critical**: The project targets:
- ROG Ally X (AMD Ryzen + Radeon) — **no NVIDIA CUDA**
- M4 MacBook Pro (Apple Silicon / MPS)

## 1. Activate the virtual environment first

```powershell
cd asr-model-comparison/backend
.\.venv\Scripts\Activate.ps1
```

## 2. Recommended: CPU-only installation (ROG Ally X / most Windows AMD users)

`faster-whisper` (based on CTranslate2) runs very well on CPU and is the easiest starting point.

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install torch CPU (stable, no CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install faster-whisper (this is the important one for Whisper models)
pip install faster-whisper>=1.0.3

# Audio utilities
pip install pydub
```

After this, you can already use all Whisper variants (tiny ~ large-v3-turbo) via the API.

## 3. Optional but recommended for development

```powershell
pip install psutil
```

## 4. For Qwen3-ASR and Voxtral (future)

These require the full `transformers` + torch stack and are heavier:

```powershell
pip install transformers torchaudio
# Then follow official Qwen3-ASR and Voxtral installation instructions
```

**Warning**: Qwen3-ASR 1.7B and Voxtral 4B will use significant RAM/VRAM even on CPU. Start with Whisper only.

## 5. First-time model download

When you call `/api/transcribe` with a Whisper model for the first time, it will automatically download the model from Hugging Face (e.g. `tiny` is ~150MB, `large-v3-turbo` is larger).

Models are cached in:
- Windows: `C:\Users\<you>\.cache\huggingface\hub` or wherever `faster-whisper` puts them.

This is expected and can take several minutes on first run.

## 6. Verification

After installation, run:

```powershell
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
```

Then start the server and test with a real audio file via the API.

## 7. Apple Silicon (MPS) users

```bash
pip install torch torchvision torchaudio
# MPS is enabled automatically in recent PyTorch + faster-whisper
```

## 8. Troubleshooting

- **Out of memory on ROG Ally X**: Start with `whisper-tiny` or `whisper-small`.
- **Very slow first inference**: Normal — model is being loaded and possibly quantized on the fly.
- **"CUDA not available"**: This is expected on AMD hardware. The CPU path is used automatically.

See the main README for how to run the server.

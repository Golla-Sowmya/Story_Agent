# API Setup Guide for MM-StoryAgent

## Required API Keys

Since the original Dashscope/Qwen API doesn't support India, this setup uses Hugging Face APIs which work globally.

### 1. Hugging Face API Key (Required)

**Get your FREE API key:**
1. Go to [https://huggingface.co/](https://huggingface.co/)
2. Create a free account
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permissions
5. Copy the token

**Set the environment variable:**
```bash
# Linux/Mac
export HUGGINGFACE_API_KEY="your_token_here"

# Windows
set HUGGINGFACE_API_KEY=your_token_here
```

### 2. Speech Synthesis (Optional - for voice generation)

If you want voice synthesis, you'll need Aliyun (Alibaba Cloud) credentials:
- `ALIYUN_ACCESS_KEY_ID`
- `ALIYUN_ACCESS_KEY_SECRET`
- `ALIYUN_APP_KEY`

## Fallback Mode

The system is designed to work even without API keys:
- **With API keys**: Full functionality with AI-generated content
- **Without API keys**: Simple rule-based fallback responses

## Running the App

```bash
# With API key
export HUGGINGFACE_API_KEY="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"
streamlit run app.py

# Without API key (fallback mode)
streamlit run app.py
```

## Supported Models

The system uses these HuggingFace models:
- **Text Generation**: `microsoft/DialoGPT-large` (default)
- **Image Generation**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Music Generation**: `facebook/musicgen-medium`
- **Sound Generation**: `cvssp/audioldm2`

All models are accessed via Hugging Face Inference API, so no local installation required!
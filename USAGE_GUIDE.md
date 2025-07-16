# MM-StoryAgent Enhanced Usage Guide

## 🌟 New Features

This enhanced version of MM-StoryAgent includes three major new capabilities:

### 1. 🌐 Multilingual Text Processing
- **Input stories in any language** (Telugu, Hindi, Marathi, Tamil, etc.)
- **Automatic language detection and translation** to English
- **Character extraction and highlighting** from multilingual text
- **Story chunking** for better processing

### 2. 🎤 Audio Input Support
- **Speech-to-text transcription** using OpenAI Whisper
- **Multi-language audio support** with automatic translation
- **High-quality transcription** with various model sizes
- **Audio format support**: MP3, WAV, FLAC, M4A, OGG

### 3. 🖼️ Reference Image Integration
- **Upload character reference images** for consistency
- **Automatic character feature extraction** from images
- **Consistent character appearance** across all generated images
- **Multiple reference images** per story

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Option 2: Command Line Interface
```bash
# For multilingual text
python run.py -c configs/multilingual_story_agent.yaml

# For audio input
python run.py -c configs/audio_input_story_agent.yaml

# For reference images
python run.py -c configs/reference_image_story_agent.yaml
```

## 📖 Detailed Usage

### Multilingual Text Processing

#### Example 1: Hindi Story
```yaml
multilingual_text:
    tool: multilingual_text_processor
    cfg:
        llm: qwen
        chunk_size: 500
        max_chunks: 10
    params:
        text: "एक बार की बात है, एक छोटे से गाँव में राजा नाम का एक बहुत बहादुर लड़का रहता था।"
        output_format: "story_topic"
```

#### Example 2: Telugu Story
```yaml
multilingual_text:
    tool: multilingual_text_processor
    cfg:
        llm: qwen
    params:
        text: "ఒకప్పుడు ఒక చిన్న గ్రామంలో రాజా అనే చాలా ధైర్యవంతుడైన అబ్బాయి ఉండేవాడు।"
        output_format: "story_topic"
```

#### Features:
- **Automatic language detection**
- **Character extraction with descriptions**
- **Story theme and setting identification**
- **Translation to English for processing**

### Audio Input Processing

#### Configuration:
```yaml
audio_input:
    tool: whisper_stt
    cfg:
        model_name: base  # Options: tiny, base, small, medium, large
        device: cuda
        language: null  # Auto-detect
        translate_to_english: true
    params:
        audio_path: "path/to/story.wav"
        output_format: "story_topic"
```

#### Supported Audio Formats:
- MP3, WAV, FLAC, M4A, OGG
- Any sample rate (automatically converted)
- Mono and stereo audio

#### Model Sizes:
- **tiny**: Fastest, lower quality
- **base**: Good balance of speed and quality
- **small**: Better quality, slower
- **medium**: High quality, requires more memory
- **large**: Highest quality, slowest

### Reference Image Integration

#### Configuration:
```yaml
image_generation:
    tool: story_diffusion_t2i
    cfg:
        reference_processor: reference_image_processor
    params:
        reference_images:
            - image_path: "path/to/character1.jpg"
              character_name: "Alice"
            - image_path: "path/to/character2.jpg"
              character_name: "Bob"
```

#### Supported Image Formats:
- JPG, JPEG, PNG, WebP
- Any resolution (automatically resized)
- RGB and RGBA images

#### Features:
- **Automatic character feature extraction**
- **Art style detection**
- **Color palette analysis**
- **Consistent character appearance**

## 🎨 Web Interface Features

### Story Input Options:
1. **📝 English Text**: Traditional text input
2. **🌐 Multilingual Text**: Input in any language
3. **🎤 Audio Recording**: Upload audio files

### Character Consistency:
- **Upload reference images** for main characters
- **Automatic character extraction** from text
- **Character cards** showing extracted information

### Generation Options:
- **Multiple art styles**: Storybook, Anime, Disney, etc.
- **Adjustable parameters**: Guidance scale, random seed
- **Progress tracking**: Real-time generation status

### Output Features:
- **Generated video** with download option
- **Individual images** for each page
- **Character information** display
- **Story metadata** showing original language, theme, setting

## 🔧 Advanced Configuration

### Custom LLM Configuration:
```yaml
story_writer:
    tool: qa_outline_story_writer
    cfg:
        llm: qwen  # or other supported LLMs
        max_conv_turns: 3
        num_outline: 4
        temperature: 0.5
```

### Custom Image Settings:
```yaml
image_generation:
    cfg:
        height: 512
        width: 1024
        model_name: stabilityai/stable-diffusion-xl-base-1.0
        id_length: 2
        reference_processor: reference_image_processor
```

### Custom Audio Settings:
```yaml
audio_input:
    cfg:
        model_name: base
        device: cuda
        translate_to_english: true
```

## 🐛 Troubleshooting

### Common Issues:

#### 1. Audio Processing Fails
```bash
# Install additional audio dependencies
pip install pydub ffmpeg-python

# Or use system ffmpeg
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

#### 2. GPU Memory Issues
```yaml
# Use smaller models
audio_input:
    cfg:
        model_name: tiny  # Instead of base/large

image_generation:
    cfg:
        id_length: 1  # Reduce consistency length
```

#### 3. Language Detection Issues
```yaml
# Specify language explicitly
audio_input:
    cfg:
        language: "hi"  # Hindi
        # or "te" for Telugu, "mr" for Marathi
```

#### 4. Character Extraction Issues
```yaml
# Use simple processor for basic needs
image_generation:
    cfg:
        reference_processor: simple_reference_processor
```

## 📊 Performance Optimization

### For Better Speed:
1. **Use smaller Whisper models** (tiny, base)
2. **Reduce image resolution** (256x512 instead of 512x1024)
3. **Limit story length** (max_chunks: 5)
4. **Use faster-whisper** instead of openai-whisper

### For Better Quality:
1. **Use larger Whisper models** (medium, large)
2. **Increase image resolution** (768x1536)
3. **More story development rounds** (max_conv_turns: 5)
4. **Higher guidance scale** (10.0-15.0)

## 📝 Example Workflows

### Workflow 1: Multilingual Story to Video
1. **Input**: Hindi text story
2. **Process**: Language detection → Translation → Character extraction
3. **Generate**: Story outline → Images → Audio → Video
4. **Output**: Complete story video with character consistency

### Workflow 2: Audio Recording to Story
1. **Input**: Audio file in any language
2. **Process**: Transcription → Translation → Story generation
3. **Generate**: Images → Speech → Music → Video
4. **Output**: Professional story video

### Workflow 3: Reference-Based Character Story
1. **Input**: Text story + Character reference images
2. **Process**: Character feature extraction → Story processing
3. **Generate**: Consistent character images → Complete video
4. **Output**: Story with visually consistent characters

## 🔮 Future Enhancements

Planned features for future versions:
- **Real-time audio recording** in web interface
- **Multiple language TTS** support
- **Advanced character pose control**
- **Story collaboration features**
- **Mobile app interface**

## 🤝 Contributing

To add new features:
1. **Create new agents** in `mm_story_agent/modality_agents/`
2. **Register with decorator**: `@register_tool("agent_name")`
3. **Update configuration** examples
4. **Add web interface** components
5. **Update documentation**

## 📞 Support

For issues or questions:
1. **Check troubleshooting** section above
2. **Review example configurations**
3. **Test with simpler settings** first
4. **Provide error logs** when reporting issues
import streamlit as st
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import uuid

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, manually load .env
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

from mm_story_agent import MMStoryAgent


# Page configuration
st.set_page_config(
    page_title="MM-StoryAgent - Multilingual Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .character-card {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 3px solid #20c997;
    }
    
    .success-message {
        padding: 15px;
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
        margin: 10px 0;
    }
    
    .error-message {
        padding: 15px;
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_story' not in st.session_state:
    st.session_state.generated_story = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'characters' not in st.session_state:
    st.session_state.characters = []
if 'story_metadata' not in st.session_state:
    st.session_state.story_metadata = {}

def save_uploaded_file(uploaded_file, directory: str) -> str:
    """Save uploaded file to temporary directory and return path."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def create_config_from_inputs(
    story_input: str,
    input_type: str,
    audio_file: Optional[str] = None,
    reference_images: Optional[List[str]] = None,
    style_name: str = "Storybook",
    guidance_scale: float = 5.0,
    seed: int = 2047,
    generation_mode: str = "complete_video",
    voice_selection: str = "longyuan",
    include_music: bool = True,
    include_sound_effects: bool = True,
    music_duration: float = 30.0
) -> Dict:
    """Create configuration dictionary from user inputs."""
    
    # Generate unique story directory
    story_id = str(uuid.uuid4())[:8]
    story_dir = f"generated_stories/story_{story_id}"
    
    config = {
        "story_dir": story_dir,
        "generation_mode": generation_mode,
        "story_writer": {
            "tool": "qa_outline_story_writer",
            "cfg": {
                "max_conv_turns": 3,
                "num_outline": 4,
                "temperature": 0.5,
                "llm": "qwen"
            },
            "params": {
                "story_topic": story_input,
                "main_role": "(no main role specified)",
                "scene": "(no scene specified)"
            }
        }
    }
    
    # Add modality-specific configurations based on generation mode
    if generation_mode in ["complete_video", "image_only"]:
        config["image_generation"] = {
            "tool": "story_diffusion_t2i",
            "cfg": {
                "num_turns": 3,
                "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
                "id_length": 2,
                "height": 512,
                "width": 1024,
                "llm": "qwen"
            },
            "params": {
                "seed": seed,
                "guidance_scale": guidance_scale,
                "style_name": style_name
            }
        }
    
    if generation_mode in ["complete_video", "audio_only"]:
        if include_sound_effects:
            config["sound_generation"] = {
                "tool": "audioldm2_t2a",
                "cfg": {
                    "num_turns": 3,
                    "device": "cuda",
                    "sample_rate": 16000
                },
                "params": {
                    "guidance_scale": 3.5,
                    "seed": 0,
                    "ddim_steps": 200,
                    "n_candidate_per_text": 3
                }
            }
        
        config["speech_generation"] = {
            "tool": "cosyvoice_tts",
            "cfg": {
                "sample_rate": 16000
            },
            "params": {
                "voice": voice_selection
            }
        }
        
        if include_music:
            config["music_generation"] = {
                "tool": "musicgen_t2m",
                "cfg": {
                    "llm_type": "qwen",
                    "num_turns": 3,
                    "device": "cuda"
                },
                "params": {
                    "duration": music_duration
                }
            }
    
    # Only add video composition for complete video mode
    if generation_mode == "complete_video":
        config["video_compose"] = {
            "tool": "slideshow_video_compose",
            "cfg": {},
            "params": {
                "height": 512,
                "width": 1024,
                "story_dir": story_dir,
                "fps": 8,
                "audio_sample_rate": 16000,
                "audio_codec": "mp3",
                "caption": {
                    "font": "resources/font/msyh.ttf",
                    "fontsize": 32,
                    "color": "white",
                    "max_length": 50
                },
                "slideshow_effect": {
                    "bg_speech_ratio": 0.6,
                    "sound_volume": 0.6,
                    "music_volume": 0.5,
                    "fade_duration": 0.8,
                    "slide_duration": 0.4,
                    "zoom_speed": 0.5,
                    "move_ratio": 0.9
                }
            }
        }
    
    # Add input-specific configuration
    if input_type == "multilingual_text":
        config["multilingual_text"] = {
            "tool": "multilingual_text_processor",
            "cfg": {
                "llm": "qwen",
                "chunk_size": 500,
                "max_chunks": 10
            },
            "params": {
                "text": story_input,
                "output_format": "story_topic"
            }
        }
    
    elif input_type == "audio" and audio_file:
        config["audio_input"] = {
            "tool": "whisper_stt",
            "cfg": {
                "model_name": "base",
                "device": "cuda",
                "language": None,
                "translate_to_english": True
            },
            "params": {
                "audio_path": audio_file,
                "output_format": "story_topic"
            }
        }
    
    # Add reference images if provided
    if reference_images:
        config["image_generation"]["params"]["reference_images"] = [
            {"image_path": img_path, "character_name": f"character_{i+1}"}
            for i, img_path in enumerate(reference_images)
        ]
    
    return config

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 MM-StoryAgent</h1>
        <p>Multilingual Story Generator with Character Consistency</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Status
    hf_key = os.environ.get('HUGGINGFACE_API_KEY')
    if hf_key:
        st.success("✅ Hugging Face API key detected - Full functionality available")
    else:
        st.warning("⚠️ No Hugging Face API key found - Running in fallback mode. See API_SETUP.md for setup instructions.")
        with st.expander("How to get a FREE Hugging Face API key"):
            st.markdown("""
            1. Go to [https://huggingface.co/](https://huggingface.co/)
            2. Create a free account
            3. Go to Settings → Access Tokens
            4. Create a new token with "Read" permissions
            5. Set environment variable: `export HUGGINGFACE_API_KEY="your_token_here"`
            6. Restart the application
            
            **The app works without API keys but with limited functionality!**
            """)
    
    # Generation Mode Selection (moved before sidebar)
    st.header("🎯 Choose Generation Mode")
    
    generation_mode = st.radio(
        "Select what you want to generate:",
        ["complete_video", "image_only", "audio_only", "text_only"],
        format_func=lambda x: {
            "complete_video": "🎬 Complete Story Video (Images + Audio + Speech + Music)",
            "image_only": "🖼️ Story Images Only",
            "audio_only": "🎵 Story Audio Only (Speech + Music + Sound)",
            "text_only": "📝 Story Text Only"
        }[x],
        index=0
    )
    
    # Mode-specific information
    mode_info = {
        "complete_video": "Generate a complete story video with images, narration, background music, and sound effects.",
        "image_only": "Generate only story illustrations with character consistency and your chosen art style.",
        "audio_only": "Generate only audio content: narrated speech, background music, and sound effects.",
        "text_only": "Generate only the story text content based on your input."
    }
    
    st.info(f"ℹ️ {mode_info[generation_mode]}")
    
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode-specific options
        if generation_mode in ["complete_video", "image_only"]:
            # Style selection for image generation
            style_name = st.selectbox(
                "Art Style",
                ["Storybook", "Japanese Anime", "Digital/Oil Painting", "Pixar/Disney Character", 
                 "Photographic", "Comic book", "Line art", "Black and White Film Noir", "Isometric Rooms"],
                index=0
            )
        else:
            style_name = "Storybook"  # Default for non-image modes
        
        if generation_mode in ["complete_video", "audio_only"]:
            # Audio-specific options
            st.subheader("🎵 Audio Options")
            voice_selection = st.selectbox(
                "Voice for Narration",
                ["longyuan", "female_voice", "male_voice"],
                index=0
            )
            
            include_music = st.checkbox("Include Background Music", value=True)
            include_sound_effects = st.checkbox("Include Sound Effects", value=True)
            
            if include_music:
                music_duration = st.slider("Background Music Duration (seconds)", 10.0, 60.0, 30.0, 5.0)
            else:
                music_duration = 30.0
        else:
            voice_selection = "longyuan"
            include_music = True
            include_sound_effects = True
            music_duration = 30.0
        
        # Generation parameters
        st.subheader("Generation Parameters")
        if generation_mode in ["complete_video", "image_only"]:
            guidance_scale = st.slider("Image Guidance Scale", 1.0, 20.0, 5.0, 0.5)
        else:
            guidance_scale = 5.0
        
        seed = st.number_input("Random Seed", 0, 100000, 2047)
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_conv_turns = st.number_input("Story Development Rounds", 1, 10, 3)
            num_outline = st.number_input("Story Chapters", 2, 10, 4)
            temperature = st.slider("Creativity Level", 0.1, 2.0, 0.5, 0.1)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Story Input")
        
        # Input type selection
        input_type = st.radio(
            "Choose input type:",
            ["text", "multilingual_text", "audio"],
            format_func=lambda x: {
                "text": "📝 English Text",
                "multilingual_text": "🌐 Multilingual Text",
                "audio": "🎤 Audio Recording"
            }[x]
        )
        
        story_input = ""
        audio_file = None
        
        if input_type == "text":
            story_input = st.text_area(
                "Enter your story idea in English:",
                placeholder="e.g., A young wizard discovers a magical book that can bring drawings to life...",
                height=150
            )
        
        elif input_type == "multilingual_text":
            st.info("💡 You can enter text in any language (Telugu, Hindi, Marathi, etc.). The system will automatically translate and extract characters.")
            story_input = st.text_area(
                "Enter your story in any language:",
                placeholder="आपकी कहानी यहाँ लिखें... / మీ కథను ఇక్కడ రాయండి... / तुमची कथा इथे लिहा...",
                height=150
            )
        
        elif input_type == "audio":
            st.info("🎤 Upload an audio file containing your story. The system will transcribe it automatically.")
            uploaded_audio = st.file_uploader(
                "Upload audio file",
                type=['mp3', 'wav', 'flac', 'm4a', 'ogg']
            )
            
            if uploaded_audio:
                # Save uploaded audio file
                temp_dir = tempfile.mkdtemp()
                audio_file = save_uploaded_file(uploaded_audio, temp_dir)
                st.audio(uploaded_audio)
                st.success(f"✅ Audio file uploaded: {uploaded_audio.name}")
        
        # Reference images section (only for modes that generate images)
        if generation_mode in ["complete_video", "image_only"]:
            st.header("🖼️ Reference Images (Optional)")
            st.info("Upload reference images of characters to maintain consistency across the story.")
            
            uploaded_images = st.file_uploader(
                "Upload character reference images",
                type=['jpg', 'jpeg', 'png', 'webp'],
                accept_multiple_files=True
            )
        else:
            uploaded_images = None
        
        reference_images = []
        if uploaded_images:
            cols = st.columns(min(len(uploaded_images), 3))
            temp_dir = tempfile.mkdtemp()
            
            for i, img in enumerate(uploaded_images):
                col_idx = i % 3
                with cols[col_idx]:
                    st.image(img, caption=f"Character {i+1}", use_column_width=True)
                    
                    # Save image
                    img_path = save_uploaded_file(img, temp_dir)
                    reference_images.append(img_path)
            
            st.success(f"✅ {len(reference_images)} reference images uploaded")
    
    with col2:
        st.header("🎯 Features")
        
        # Mode-specific feature cards
        if generation_mode == "complete_video":
            st.markdown("""
            <div class="feature-card">
                <h4>🎬 Complete Video Generation</h4>
                <p>Creates images, speech, music, and sound effects</p>
            </div>
            
            <div class="feature-card">
                <h4>🎨 Character Consistency</h4>
                <p>Upload reference images for consistent character appearance</p>
            </div>
            
            <div class="feature-card">
                <h4>🌐 Multilingual Support</h4>
                <p>Input stories in Telugu, Hindi, Marathi, or any language</p>
            </div>
            """, unsafe_allow_html=True)
        elif generation_mode == "image_only":
            st.markdown("""
            <div class="feature-card">
                <h4>🎨 Story Illustrations</h4>
                <p>Generate consistent character images across story pages</p>
            </div>
            
            <div class="feature-card">
                <h4>🎭 Multiple Art Styles</h4>
                <p>Choose from storybook, anime, Disney, and more</p>
            </div>
            """, unsafe_allow_html=True)
        elif generation_mode == "audio_only":
            st.markdown("""
            <div class="feature-card">
                <h4>🎵 Complete Audio Experience</h4>
                <p>Narration, background music, and sound effects</p>
            </div>
            
            <div class="feature-card">
                <h4>🗣️ Voice Selection</h4>
                <p>Choose from multiple voice options for narration</p>
            </div>
            """, unsafe_allow_html=True)
        else:  # text_only
            st.markdown("""
            <div class="feature-card">
                <h4>📝 Story Writing</h4>
                <p>AI-powered story generation with multiple chapters</p>
            </div>
            
            <div class="feature-card">
                <h4>🌐 Multilingual Support</h4>
                <p>Input and output in multiple languages</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Generate button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Dynamic button text based on generation mode
        button_text = {
            "complete_video": "🚀 Generate Story Video",
            "image_only": "🖼️ Generate Story Images",
            "audio_only": "🎵 Generate Story Audio",
            "text_only": "📝 Generate Story Text"
        }
        
        generate_button = st.button(
            button_text[generation_mode],
            type="primary",
            disabled=st.session_state.processing or not (story_input or audio_file),
            use_container_width=True
        )
    
    # Generation process
    if generate_button:
        if not story_input and not audio_file:
            st.error("❌ Please provide story input (text or audio)")
            return
        
        st.session_state.processing = True
        
        # Create progress indicators
        progress_container = st.container()
        with progress_container:
            st.info("🔄 Starting story generation...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Create configuration based on generation mode
            config = create_config_from_inputs(
                story_input=story_input,
                input_type=input_type,
                audio_file=audio_file,
                reference_images=reference_images,
                style_name=style_name,
                guidance_scale=guidance_scale,
                seed=seed,
                generation_mode=generation_mode,
                voice_selection=voice_selection,
                include_music=include_music,
                include_sound_effects=include_sound_effects,
                music_duration=music_duration
            )
            
            # Initialize and run MM-StoryAgent
            status_text.text("🤖 Initializing MM-StoryAgent...")
            progress_bar.progress(10)
            
            agent = MMStoryAgent()
            
            status_text.text("📝 Generating story content...")
            progress_bar.progress(25)
            
            # Run the agent
            agent.call(config)
            
            progress_bar.progress(100)
            status_text.text("✅ Story generation complete!")
            
            # Store results
            st.session_state.generated_story = config["story_dir"]
            st.session_state.characters = getattr(agent, 'extracted_characters', [])
            st.session_state.story_metadata = getattr(agent, 'story_metadata', {})
            
            st.success("🎉 Story video generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            st.exception(e)
        
        finally:
            st.session_state.processing = False
    
    # Display results
    if st.session_state.generated_story:
        st.markdown("---")
        st.header("📊 Generation Results")
        
        story_dir = Path(st.session_state.generated_story)
        
        # Display extracted characters
        if st.session_state.characters:
            st.subheader("🎭 Extracted Characters")
            
            for char in st.session_state.characters:
                st.markdown(f"""
                <div class="character-card">
                    <h4>{char.get('name', 'Unknown')}</h4>
                    <p><strong>Role:</strong> {char.get('role', 'Unknown')}</p>
                    <p><strong>Description:</strong> {char.get('description', 'No description')}</p>
                    <p><strong>Key Traits:</strong> {', '.join(char.get('key_traits', []))}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display story metadata
        if st.session_state.story_metadata:
            st.subheader("📋 Story Metadata")
            metadata = st.session_state.story_metadata
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
                st.write(f"**Original Language:** {metadata.get('original_language', 'Unknown')}")
            
            with col2:
                st.write(f"**Theme:** {metadata.get('theme', 'Not specified')}")
                st.write(f"**Setting:** {metadata.get('setting', 'Not specified')}")
        
        # Display generated files
        st.subheader("📁 Generated Files")
        
        # Check for generated video
        video_files = list(story_dir.glob("*.mp4"))
        if video_files:
            st.video(str(video_files[0]))
            
            # Download button
            with open(video_files[0], "rb") as f:
                st.download_button(
                    label="⬇️ Download Story Video",
                    data=f.read(),
                    file_name=f"story_video_{story_dir.name}.mp4",
                    mime="video/mp4"
                )
        
        # Display generated images
        image_dir = story_dir / "image"
        if image_dir.exists():
            image_files = list(image_dir.glob("*.png"))
            if image_files:
                st.subheader("🖼️ Generated Images")
                
                cols = st.columns(min(len(image_files), 3))
                for i, img_file in enumerate(image_files):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.image(str(img_file), caption=f"Page {i+1}", use_column_width=True)
        
        # Display script data
        script_file = story_dir / "script_data.json"
        if script_file.exists():
            with st.expander("📜 View Story Script"):
                with open(script_file, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                    st.json(script_data)

if __name__ == "__main__":
    main()
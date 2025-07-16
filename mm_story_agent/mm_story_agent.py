import time
import json
from pathlib import Path
from typing import Dict, Union, Optional
import multiprocessing as mp

from .base import init_tool_instance


class MMStoryAgent:

    def __init__(self) -> None:
        self.modalities = ["image", "sound", "speech", "music"]

    def call_modality_agent(self, modality, agent, params, return_dict):
        result = agent.call(params)
        return_dict[modality] = result

    def preprocess_audio_input(self, config) -> Optional[Dict]:
        """
        Preprocess audio input to extract story topic using speech-to-text.
        
        Args:
            config: Configuration dictionary that may contain audio_input section
            
        Returns:
            Extracted story data or None if no audio input
        """
        if "audio_input" not in config:
            return None
            
        audio_cfg = config["audio_input"]
        stt_agent = init_tool_instance(audio_cfg)
        
        # Process audio input
        result = stt_agent.call(audio_cfg["params"])
        
        if result.get("success", False):
            return {
                "story_topic": result.get("story_topic", result.get("text", "")),
                "source": "audio",
                "language": result.get("language", "unknown")
            }
        else:
            print(f"Audio transcription failed: {result.get('error', 'Unknown error')}")
            return None

    def preprocess_multilingual_text(self, config) -> Optional[Dict]:
        """
        Preprocess multilingual text input for story generation.
        
        Args:
            config: Configuration dictionary that may contain multilingual_text section
            
        Returns:
            Processed story data or None if no multilingual text input
        """
        if "multilingual_text" not in config:
            return None
            
        text_cfg = config["multilingual_text"]
        text_processor = init_tool_instance(text_cfg)
        
        # Process multilingual text
        result = text_processor.call(text_cfg["params"])
        
        if result.get("success", False):
            return {
                "story_topic": result.get("story_topic", ""),
                "characters": result.get("characters", []),
                "story_theme": result.get("story_theme", ""),
                "setting": result.get("setting", ""),
                "source": "multilingual_text",
                "original_language": result.get("original_language", "unknown"),
                "chunks": result.get("chunks", [])
            }
        else:
            print(f"Multilingual text processing failed: {result.get('error', 'Unknown error')}")
            return None

    def write_story(self, config):
        cfg = config["story_writer"]
        
        # Check for different input types
        audio_data = self.preprocess_audio_input(config)
        multilingual_data = self.preprocess_multilingual_text(config)
        
        # Store character information for later use
        self.extracted_characters = []
        self.story_metadata = {}
        
        if multilingual_data:
            # Use multilingual text as primary input
            if "story_topic" in cfg["params"]:
                # Combine existing topic with multilingual input
                original_topic = cfg["params"]["story_topic"]
                cfg["params"]["story_topic"] = f"{original_topic}. Story context: {multilingual_data['story_topic']}"
            else:
                # Use multilingual text as main story topic
                cfg["params"]["story_topic"] = multilingual_data["story_topic"]
            
            # Store character and metadata information
            self.extracted_characters = multilingual_data.get("characters", [])
            self.story_metadata = {
                "theme": multilingual_data.get("story_theme", ""),
                "setting": multilingual_data.get("setting", ""),
                "original_language": multilingual_data.get("original_language", ""),
                "source": "multilingual_text"
            }
            
            # Add character information to story generation
            if self.extracted_characters:
                char_descriptions = []
                for char in self.extracted_characters:
                    char_desc = f"{char['name']}: {char['description']} (Role: {char['role']})"
                    char_descriptions.append(char_desc)
                
                cfg["params"]["main_role"] = char_descriptions[0] if char_descriptions else "(no main role specified)"
                cfg["params"]["characters"] = "; ".join(char_descriptions)
        
        elif audio_data:
            # Use audio input as fallback
            if "story_topic" in cfg["params"]:
                original_topic = cfg["params"]["story_topic"]
                cfg["params"]["story_topic"] = f"{original_topic}. Additional context from audio: {audio_data['story_topic']}"
            else:
                cfg["params"]["story_topic"] = audio_data["story_topic"]
            
            self.story_metadata = {
                "source": "audio",
                "language": audio_data.get("language", "unknown")
            }
        
        story_writer = init_tool_instance(cfg)
        pages = story_writer.call(cfg["params"])
        return pages
    
    def generate_modality_assets(self, config, pages):
        script_data = {"pages": [{"story": page} for page in pages]}
        
        # Add character and metadata information to script data
        if hasattr(self, 'extracted_characters') and self.extracted_characters:
            script_data["characters"] = self.extracted_characters
        
        if hasattr(self, 'story_metadata') and self.story_metadata:
            script_data["metadata"] = self.story_metadata
        
        story_dir = Path(config["story_dir"])
        generation_mode = config.get("generation_mode", "complete_video")
        
        # Determine which modalities to generate based on mode
        active_modalities = []
        if generation_mode in ["complete_video", "image_only"]:
            active_modalities.append("image")
        if generation_mode in ["complete_video", "audio_only"]:
            if "speech_generation" in config:
                active_modalities.append("speech")
            if "sound_generation" in config:
                active_modalities.append("sound")
            if "music_generation" in config:
                active_modalities.append("music")

        # Create directories for active modalities
        for modality in active_modalities:
            (story_dir / modality).mkdir(exist_ok=True, parents=True)

        agents = {}
        params = {}
        
        # Initialize agents for active modalities only
        for modality in active_modalities:
            config_key = modality + "_generation"
            if config_key in config:
                agents[modality] = init_tool_instance(config[config_key])
                params[modality] = config[config_key]["params"].copy()
                params[modality].update({
                    "pages": pages,
                    "save_path": story_dir / modality
                })

        processes = []
        return_dict = mp.Manager().dict()

        # Start processes for active modalities
        for modality in active_modalities:
            if modality in agents:
                p = mp.Process(
                    target=self.call_modality_agent,
                    args=(
                        modality,
                        agents[modality],
                        params[modality],
                        return_dict)
                    )
                processes.append(p)
                p.start()
        
        for p in processes:
            p.join()

        images = None
        for modality, result in return_dict.items():
            try:
                if modality == "image":
                    images = result["generation_results"]
                    for idx in range(len(pages)):
                        script_data["pages"][idx]["image_prompt"] = result["prompts"][idx]
                elif modality == "sound":
                    for idx in range(len(pages)):
                        script_data["pages"][idx]["sound_prompt"] = result["prompts"][idx]
                elif modality == "music":
                    script_data["music_prompt"] = result["prompt"]
            except Exception as e:
                print(f"Error occurred during generation: {e}")
        
        with open(story_dir / "script_data.json", "w") as writer:
            json.dump(script_data, writer, ensure_ascii=False, indent=4)
        
        return images
    
    def compose_storytelling_video(self, config, pages):
        video_compose_agent = init_tool_instance(config["video_compose"])
        params = config["video_compose"]["params"].copy()
        params["pages"] = pages
        video_compose_agent.call(params)

    def call(self, config):
        generation_mode = config.get("generation_mode", "complete_video")
        pages = self.write_story(config)
        images = self.generate_modality_assets(config, pages)
        
        # Only compose video for complete video mode
        if generation_mode == "complete_video" and "video_compose" in config:
            self.compose_storytelling_video(config, pages)

import os
import torch
import whisper
from typing import Dict, Optional
from pathlib import Path

from mm_story_agent.base import register_tool


@register_tool("whisper_stt")
class WhisperSTTAgent:
    """
    Speech-to-Text agent using OpenAI Whisper for multi-language transcription.
    Supports automatic language detection and translation to English.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model_name = cfg.get("model_name", "base")
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.language = cfg.get("language", None)  # Auto-detect if None
        self.translate_to_english = cfg.get("translate_to_english", True)
        
        # Load Whisper model
        self.model = whisper.load_model(self.model_name, device=self.device)
        
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcribed text and metadata
        """
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task="translate" if self.translate_to_english else "transcribe",
                verbose=False
            )
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result["segments"],
                "success": True
            }
            
        except Exception as e:
            return {
                "text": "",
                "language": "unknown",
                "segments": [],
                "success": False,
                "error": str(e)
            }
    
    def call(self, params: Dict) -> Dict:
        """
        Main call method for the speech-to-text agent.
        
        Args:
            params: Dictionary containing:
                - audio_path: Path to audio file or list of audio files
                - output_format: "text" or "story_topic" (default: "story_topic")
                
        Returns:
            Dictionary containing transcribed text and metadata
        """
        audio_path = params.get("audio_path")
        output_format = params.get("output_format", "story_topic")
        
        if not audio_path:
            return {
                "success": False,
                "error": "No audio path provided"
            }
        
        # Handle single audio file
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
            if not audio_path.exists():
                return {
                    "success": False,
                    "error": f"Audio file not found: {audio_path}"
                }
            
            result = self.transcribe_audio(str(audio_path))
            
            if result["success"]:
                # Format output based on requested format
                if output_format == "story_topic":
                    # Format as story topic for the story agent
                    formatted_text = f"Story Topic: {result['text']}"
                    if result["language"] != "en":
                        formatted_text += f" (Original language: {result['language']})"
                    
                    return {
                        "story_topic": formatted_text,
                        "raw_text": result["text"],
                        "language": result["language"],
                        "success": True
                    }
                else:
                    # Return raw text
                    return {
                        "text": result["text"],
                        "language": result["language"],
                        "success": True
                    }
            else:
                return result
        
        # Handle multiple audio files
        elif isinstance(audio_path, list):
            results = []
            combined_text = ""
            
            for i, path in enumerate(audio_path):
                path = Path(path)
                if not path.exists():
                    continue
                    
                result = self.transcribe_audio(str(path))
                if result["success"]:
                    results.append(result)
                    combined_text += f" {result['text']}"
            
            if results:
                # Combine all transcriptions
                if output_format == "story_topic":
                    formatted_text = f"Story Topic: {combined_text.strip()}"
                    return {
                        "story_topic": formatted_text,
                        "raw_text": combined_text.strip(),
                        "individual_results": results,
                        "success": True
                    }
                else:
                    return {
                        "text": combined_text.strip(),
                        "individual_results": results,
                        "success": True
                    }
            else:
                return {
                    "success": False,
                    "error": "No valid audio files found or transcription failed"
                }
        
        else:
            return {
                "success": False,
                "error": "Invalid audio_path format. Expected string or list of strings."
            }


@register_tool("fast_whisper_stt")
class FastWhisperSTTAgent:
    """
    Faster Speech-to-Text agent using faster-whisper for better performance.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model_name = cfg.get("model_name", "base")
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.language = cfg.get("language", None)
        self.translate_to_english = cfg.get("translate_to_english", True)
        
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            self.use_faster_whisper = True
        except ImportError:
            # Fall back to regular whisper if faster-whisper is not available
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.use_faster_whisper = False
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file using faster-whisper or regular whisper.
        """
        try:
            if self.use_faster_whisper:
                # Use faster-whisper
                segments, info = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task="translate" if self.translate_to_english else "transcribe"
                )
                
                # Convert segments to text
                text = " ".join([segment.text for segment in segments])
                
                return {
                    "text": text.strip(),
                    "language": info.language,
                    "success": True
                }
            else:
                # Use regular whisper
                result = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task="translate" if self.translate_to_english else "transcribe"
                )
                
                return {
                    "text": result["text"].strip(),
                    "language": result["language"],
                    "success": True
                }
                
        except Exception as e:
            return {
                "text": "",
                "language": "unknown",
                "success": False,
                "error": str(e)
            }
    
    def call(self, params: Dict) -> Dict:
        """
        Main call method - similar to WhisperSTTAgent but with faster processing.
        """
        audio_path = params.get("audio_path")
        output_format = params.get("output_format", "story_topic")
        
        if not audio_path:
            return {
                "success": False,
                "error": "No audio path provided"
            }
        
        # Handle single audio file
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
            if not audio_path.exists():
                return {
                    "success": False,
                    "error": f"Audio file not found: {audio_path}"
                }
            
            result = self.transcribe_audio(str(audio_path))
            
            if result["success"]:
                if output_format == "story_topic":
                    formatted_text = f"Story Topic: {result['text']}"
                    if result["language"] != "en":
                        formatted_text += f" (Original language: {result['language']})"
                    
                    return {
                        "story_topic": formatted_text,
                        "raw_text": result["text"],
                        "language": result["language"],
                        "success": True
                    }
                else:
                    return {
                        "text": result["text"],
                        "language": result["language"],
                        "success": True
                    }
            else:
                return result
        
        # Handle multiple audio files
        elif isinstance(audio_path, list):
            results = []
            combined_text = ""
            
            for path in audio_path:
                path = Path(path)
                if not path.exists():
                    continue
                    
                result = self.transcribe_audio(str(path))
                if result["success"]:
                    results.append(result)
                    combined_text += f" {result['text']}"
            
            if results:
                if output_format == "story_topic":
                    formatted_text = f"Story Topic: {combined_text.strip()}"
                    return {
                        "story_topic": formatted_text,
                        "raw_text": combined_text.strip(),
                        "individual_results": results,
                        "success": True
                    }
                else:
                    return {
                        "text": combined_text.strip(),
                        "individual_results": results,
                        "success": True
                    }
            else:
                return {
                    "success": False,
                    "error": "No valid audio files found or transcription failed"
                }
        
        else:
            return {
                "success": False,
                "error": "Invalid audio_path format. Expected string or list of strings."
            }
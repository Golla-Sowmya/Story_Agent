from pathlib import Path
import json
from typing import List, Union, Dict
import requests
import os
from io import BytesIO

import soundfile as sf
import numpy as np

from mm_story_agent.prompts_en import story_to_music_reviser_system, story_to_music_reviewer_system
from mm_story_agent.base import register_tool, init_tool_instance


class HuggingFaceMusicSynthesizer:
    """API-based music generation using Hugging Face Inference API"""
    
    def __init__(self,
                 model_name: str = 'facebook/musicgen-medium',
                 sample_rate: int = 16000) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.environ.get('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')}"}
    
    def call(self,
             prompt: Union[str, List[str]],
             save_path: Union[str, Path],
             duration: float = 30.0):
        """Generate music using Hugging Face API"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "duration": duration,
                "sample_rate": self.sample_rate
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            if response.status_code == 200:
                # Handle audio response
                audio_data = response.content
                with open(save_path, 'wb') as f:
                    f.write(audio_data)
            else:
                print(f"Music API Error: {response.status_code} - {response.text}")
                # Create a silent audio file as fallback
                silent_audio = np.zeros(int(self.sample_rate * duration))
                sf.write(save_path, silent_audio, self.sample_rate)
        except Exception as e:
            print(f"Music generation failed: {e}")
            # Create a silent audio file as fallback
            silent_audio = np.zeros(int(self.sample_rate * duration))
            sf.write(save_path, silent_audio, self.sample_rate)


@register_tool("musicgen_t2m")
class MusicGenAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def generate_music_prompt_from_story(
            self,
            pages: List,
        ):
        music_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_music_reviser_system,
                "track_history": False
            }
        })
        music_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_music_reviewer_system,
                "track_history": False
            }
        })

        music_prompt = ""
        review = ""
        for turn in range(self.cfg.get("max_turns", 3)):
            music_prompt, success = music_prompt_reviser.call(json.dumps({
                "story": pages,
                "previous_result": music_prompt,
                "improvement_suggestions": review,
            }, ensure_ascii=False))
            review, success = music_prompt_reviewer.call(json.dumps({
                "story_content": pages,
                "music_description": music_prompt
            }, ensure_ascii=False))
            if review == "Check passed.":
                break
        
        return music_prompt

    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        save_path = Path(save_path)
        music_prompt = self.generate_music_prompt_from_story(pages)
        generation_agent = HuggingFaceMusicSynthesizer(
            model_name=self.cfg.get("model_name", "facebook/musicgen-medium"),
            sample_rate=self.cfg.get("sample_rate", 16000)
        )
        generation_agent.call(
            prompt=music_prompt,
            save_path=save_path / "music.wav",
            duration=params.get("duration", 30.0),
        )
        return {
            "prompt": music_prompt,
        }
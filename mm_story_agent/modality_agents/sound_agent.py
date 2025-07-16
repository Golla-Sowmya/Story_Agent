from pathlib import Path
from typing import List, Dict
import json
import requests
import os
from io import BytesIO

import soundfile as sf
import numpy as np

from mm_story_agent.prompts_en import story_to_sound_reviser_system, story_to_sound_review_system
from mm_story_agent.base import register_tool, init_tool_instance


class HuggingFaceAudioSynthesizer:
    """API-based audio generation using Hugging Face Inference API"""

    def __init__(self,
                 model_name: str = "cvssp/audioldm2",
                 sample_rate: int = 16000) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.environ.get('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')}"}
    
    def generate_single_audio(self, prompt: str, guidance_scale: float = 3.5, 
                             ddim_steps: int = 100, seed: int = 0, 
                             audio_length_in_s: float = 10.0):
        """Generate a single audio clip"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": guidance_scale,
                "num_inference_steps": ddim_steps,
                "audio_length_in_s": audio_length_in_s,
                "seed": seed
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            if response.status_code == 200:
                # Assume the response is raw audio data
                audio_data = response.content
                # Convert to numpy array (this might need adjustment based on actual API response format)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                return audio_array
            else:
                print(f"Audio API Error: {response.status_code} - {response.text}")
                # Return silent audio as fallback
                return np.zeros(int(self.sample_rate * audio_length_in_s))
        except Exception as e:
            print(f"Audio generation failed: {e}")
            # Return silent audio as fallback
            return np.zeros(int(self.sample_rate * audio_length_in_s))
    
    def call(
        self,
        prompts: List[str],
        n_candidate_per_text: int = 3,
        seed: int = 0,
        guidance_scale: float = 3.5,
        ddim_steps: int = 100,
    ):
        """Generate multiple audio clips"""
        audios = []
        for i, prompt in enumerate(prompts):
            audio_seed = seed + i
            audio = self.generate_single_audio(
                prompt=prompt,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                seed=audio_seed,
                audio_length_in_s=10.0
            )
            audios.append(audio)
        
        return audios


@register_tool("audioldm2_t2a")
class AudioLDM2Agent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        sound_prompts = self.generate_sound_prompt_from_story(pages,)
        save_paths = []
        forward_prompts = []
        save_path = Path(save_path)
        for idx in range(len(pages)):
            if sound_prompts[idx] != "No sounds.":
                save_paths.append(save_path / f"p{idx + 1}.wav")
                forward_prompts.append(sound_prompts[idx])
        
        generation_agent = HuggingFaceAudioSynthesizer(
            model_name=self.cfg.get("model_name", "cvssp/audioldm2"),
            sample_rate=self.cfg.get("sample_rate", 16000)
        )
        if len(forward_prompts) > 0:
            sounds = generation_agent.call(
                forward_prompts,
                n_candidate_per_text=params.get("n_candidate_per_text", 3),
                seed=params.get("seed", 0),
                guidance_scale=params.get("guidance_scale", 3.5),
                ddim_steps=params.get("ddim_steps", 100),
            )
            for sound, path in zip(sounds, save_paths):
                sf.write(path.__str__(), sound, self.cfg["sample_rate"])
        return {
            "prompts": sound_prompts,
        }

    def generate_sound_prompt_from_story(
            self,
            pages: List,
        ):
        sound_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_sound_reviser_system,
                "track_history": False
            }
        })
        sound_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_sound_review_system,
                "track_history": False
            }
        })
        num_turns = self.cfg.get("num_turns", 3)

        sound_prompts = []
        for page in pages:
            review = ""
            sound_prompt = ""
            for turn in range(num_turns):
                sound_prompt, success = sound_prompt_reviser.call(json.dumps({
                    "story": page,
                    "previous_result": sound_prompt,
                    "improvement_suggestions": review,
                }, ensure_ascii=False))
                if sound_prompt.startswith("Sound description:"):
                    sound_prompt = sound_prompt[len("Sound description:"):]
                review, success = sound_prompt_reviewer.call(json.dumps({
                    "story": page,
                    "sound_description": sound_prompt
                }, ensure_ascii=False))
                if review == "Check passed.":
                    break
                # else:
                    # print(review)
            sound_prompts.append(sound_prompt)

        return sound_prompts


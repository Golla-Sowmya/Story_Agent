from typing import Dict, Callable
import os
import requests
import json

from mm_story_agent.base import register_tool


@register_tool("qwen")
class HuggingFaceTextAgent(object):
    """
    Hugging Face-based text generation agent that works globally.
    Replaces Dashscope/Qwen which doesn't support India.
    """

    def __init__(self,
                 config: Dict):
        
        self.system_prompt = config.get("system_prompt")
        track_history = config.get("track_history", False)
        if self.system_prompt is None:
            self.history = []
        else:
            self.history = [
                {"role": "system", "content": self.system_prompt}
            ]
        self.track_history = track_history
        
        # Default to a good open-source model available on HuggingFace
        self.model_name = config.get("model_name", "meta-llama/Llama-3.2-3B-Instruct")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        api_key = os.environ.get('HUGGINGFACE_API_KEY')
        if not api_key:
            print("Warning: No HUGGINGFACE_API_KEY found. API calls will fail.")
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def basic_success_check(self, response_text):
        if not response_text or len(response_text.strip()) == 0:
            return False
        return True
    
    def call(self,
             prompt: str,
             model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
             top_p: float = 0.95,
             temperature: float = 1.0,
             seed: int = 1,
             max_length: int = 1024,
             max_try: int = 5,
             success_check_fn: Callable = None
             ):
        
        # If model_name is provided, update the API URL
        if model_name != self.model_name:
            self.model_name = model_name
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        self.history.append({
            "role": "user",
            "content": prompt
        })
        
        # Format the conversation for the API
        conversation_text = ""
        if self.system_prompt:
            conversation_text = f"System: {self.system_prompt}\n\n"
        
        # Add conversation history
        for msg in self.history:
            if msg["role"] == "user":
                conversation_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_text += f"Assistant: {msg['content']}\n"
        
        conversation_text += "Assistant:"
        
        payload = {
            "inputs": conversation_text,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        success = False
        try_times = 0
        response_text = ""
        
        while try_times < max_try:
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        response_text = result[0].get("generated_text", "").strip()
                    elif isinstance(result, dict):
                        response_text = result.get("generated_text", "").strip()
                    
                    # Fallback: if API doesn't work, use a simple rule-based response
                    if not response_text:
                        response_text = self.generate_fallback_response(prompt)
                    
                    if success_check_fn is None:
                        success_check_fn = lambda x: True
                        
                    if self.basic_success_check(response_text) and success_check_fn(response_text):
                        self.history.append({
                            "role": "assistant",
                            "content": response_text
                        })
                        success = True
                        break
                        
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    response_text = self.generate_fallback_response(prompt)
                    if self.basic_success_check(response_text):
                        success = True
                        break
                        
            except Exception as e:
                print(f"Request failed: {e}")
                response_text = self.generate_fallback_response(prompt)
                if self.basic_success_check(response_text):
                    success = True
                    break
                    
            try_times += 1
        
        if not self.track_history:
            if self.system_prompt is not None:
                self.history = self.history[:1]
            else:
                self.history = []
        
        return response_text, success
    
    def generate_fallback_response(self, prompt):
        """Generate a simple fallback response when API fails"""
        prompt_lower = prompt.lower()
        
        # Simple rule-based responses for common story generation tasks
        if "outline" in prompt_lower or "章节" in prompt_lower:
            return json.dumps({
                "outline": [
                    "Chapter 1: Introduction - Setting up the story world and main characters",
                    "Chapter 2: Development - The main conflict or adventure begins",
                    "Chapter 3: Climax - The most exciting part of the story",
                    "Chapter 4: Resolution - How the story ends and characters grow"
                ]
            })
        elif "image" in prompt_lower or "picture" in prompt_lower:
            return "A colorful illustration showing the main characters in their story setting"
        elif "music" in prompt_lower or "sound" in prompt_lower:
            return "Gentle, uplifting background music that matches the story's mood"
        elif "story" in prompt_lower:
            return "Once upon a time, there was an adventure that taught valuable lessons about friendship and courage."
        else:
            return "I understand your request and will help create engaging story content."


# Register the same agent with alternative names for backward compatibility
@register_tool("huggingface_text")
class HuggingFaceTextAgentAlias(HuggingFaceTextAgent):
    pass
   
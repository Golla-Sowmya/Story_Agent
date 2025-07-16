from typing import List, Dict
import json
import os
import random
import requests
import base64
from io import BytesIO

import numpy as np
from PIL import Image

from mm_story_agent.prompts_en import role_extract_system, role_review_system, \
    story_to_image_reviser_system, story_to_image_review_system
from mm_story_agent.base import register_tool, init_tool_instance


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)








class HuggingFaceImageSynthesizer:
    """API-based image generation using Hugging Face Inference API"""
    
    def __init__(self,
                 num_pages: int,
                 height: int = 512,
                 width: int = 1024,
                 model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.num_pages = num_pages
        self.height = height
        self.width = width
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.environ.get('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')}"}
        self.styles = {
            '(No style)': ('{prompt}', ''),
            'Japanese Anime': (
                'anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. best quality, high resolution, (Anime Style, Manga Style:1.3), Low detail, sketch, concept art, line art, webtoon, manhua, hand drawn, defined lines, simple shades, minimalistic, High contrast, Linear compositions, Scalable artwork, Digital art, High Contrast Shadows',
                'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Digital/Oil Painting': (
                '{prompt} . (Extremely Detailed Oil Painting:1.2), glow effects, godrays, Hand drawn, render, 8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, Sharp focus, humorous illustration, big depth of field',
                'anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Pixar/Disney Character': (
                'Create a Disney Pixar 3D style illustration on {prompt} . The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.',
                'lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo'),
            'Photographic': (
                'cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, soft focus',
                'drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Comic book': (
                'comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed',
                'photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Line art': (
                'line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Black and White Film Noir': (
                '{prompt} . (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Isometric Rooms': (
                'Tiny cute isometric {prompt} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render',
                'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'),
            'Storybook': (
                "Cartoon style, cute illustration of {prompt}.",
                'realism, photo, realistic, lowres, bad hands, bad eyes, bad arms, bad legs, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, grayscale, noisy, sloppy, messy, grainy, ultra textured'
            )
        }
        self.negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation," \
                               "extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating" \
                               "limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
    
    def apply_style_positive(self, style_name: str, positive: str):
        p, n = self.styles.get(style_name, self.styles["(No style)"])
        return p.replace("{prompt}", positive)
    
    def generate_image(self, prompt: str, negative_prompt: str = "", guidance_scale: float = 7.5, seed: int = None):
        """Generate a single image using Hugging Face Inference API"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "height": self.height,
                "width": self.width
            }
        }
        
        if seed is not None:
            payload["parameters"]["seed"] = seed
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                return image
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                # Return a placeholder image
                return Image.new('RGB', (self.width, self.height), color='lightgray')
        except Exception as e:
            print(f"Image generation failed: {e}")
            # Return a placeholder image
            return Image.new('RGB', (self.width, self.height), color='lightgray')
    
    def call(self,
             prompts: List[str],
             style_name: str = "Storybook",
             guidance_scale: float = 7.5,
             seed: int = 2047):
        """Generate multiple images for story pages"""
        setup_seed(seed)
        images = []
        
        for i, prompt in enumerate(prompts):
            styled_prompt = self.apply_style_positive(style_name, prompt)
            image_seed = seed + i if seed else None
            image = self.generate_image(
                prompt=styled_prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=guidance_scale,
                seed=image_seed
            )
            images.append(image)
            
        return images


@register_tool("story_diffusion_t2i")
class StoryDiffusionAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    def process_reference_images(self, params: Dict) -> Dict:
        """
        Process reference images if provided.
        
        Args:
            params: Parameters that may contain reference_images or characters
            
        Returns:
            Dictionary with processed reference data
        """
        result = {"characters": []}
        
        # Check if we have reference images to process
        if "reference_images" in params:
            try:
                # Initialize reference processor
                ref_processor = init_tool_instance({
                    "tool": self.cfg.get("reference_processor", "simple_reference_processor"),
                    "cfg": {
                        "llm": self.cfg.get("llm", "qwen"),
                        "image_size": (self.cfg.get("height", 512), self.cfg.get("width", 512))
                    }
                })
                
                # Process reference images
                ref_result = ref_processor.call({"reference_images": params["reference_images"]})
                
                if ref_result.get("success", False):
                    result["characters"] = ref_result.get("references", [])
                    
            except Exception as e:
                print(f"Reference image processing failed: {e}")
        
        # Check if we have extracted characters from multilingual text
        elif "characters" in params:
            # Use characters extracted from multilingual text processing
            characters = params["characters"]
            for char in characters:
                result["characters"].append({
                    "name": char.get("name", "character"),
                    "description": char.get("description", ""),
                    "art_style": "story illustration style",
                    "key_features": char.get("key_traits", []),
                    "role": char.get("role", "character")
                })
        
        return result
        
    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        
        # Check if we have reference images or extracted characters
        reference_data = self.process_reference_images(params)
        
        # Use provided characters or extract from story
        if reference_data.get("characters"):
            role_dict = {char["name"]: char["description"] for char in reference_data["characters"]}
        else:
            role_dict = self.extract_role_from_story(pages)
        
        image_prompts = self.generate_image_prompt_from_story(pages)
        image_prompts_with_role_desc = []
        
        # Apply character descriptions to prompts
        for image_prompt in image_prompts:
            # First apply extracted characters if available
            if reference_data.get("characters"):
                for char in reference_data["characters"]:
                    char_name = char["name"].lower()
                    if char_name in image_prompt.lower():
                        # Use reference-based description
                        char_desc = f"{char['description']} ({char.get('art_style', 'consistent style')})"
                        image_prompt = image_prompt.replace(char["name"], char_desc)
            
            # Then apply traditional role descriptions
            for role, role_desc in role_dict.items():
                if role in image_prompt:
                    image_prompt = image_prompt.replace(role, role_desc)
            
            image_prompts_with_role_desc.append(image_prompt)
        
        generation_agent = HuggingFaceImageSynthesizer(
            num_pages=len(pages),
            height=self.cfg.get("height", 512),
            width=self.cfg.get("width", 1024),
            model_name=self.cfg.get("model_name", "stabilityai/stable-diffusion-xl-base-1.0")
        )
        images = generation_agent.call(
            image_prompts_with_role_desc,
            style_name=params.get("style_name", "Storybook"),
            guidance_scale=params.get("guidance_scale", 5.0),
            seed=params.get("seed", 2047)
        )
        for idx, image in enumerate(images):
            image.save(save_path / f"p{idx + 1}.png")
        
        result = {
            "prompts": image_prompts_with_role_desc,
            "generation_results": images,
        }
        
        # Add reference information if available
        if reference_data.get("characters"):
            result["reference_characters"] = reference_data["characters"]
        
        return result
        
    def extract_role_from_story(
            self,
            pages: List,
        ):
        num_turns = self.cfg.get("num_turns", 3)
        role_extractor = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": role_extract_system,
                "track_history": False
            }
        })
        role_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": role_review_system,
                "track_history": False
            }
        })
        roles = {}
        review = ""
        for turn in range(num_turns):
            roles, success = role_extractor.call(json.dumps({
                    "story_content": pages,
                    "previous_result": roles,
                    "improvement_suggestions": review,
                }, ensure_ascii=False
            ))
            roles = json.loads(roles.strip("```json").strip("```"))
            review, success = role_reviewer.call(json.dumps({
                "story_content": pages,
                "role_descriptions": roles
            }, ensure_ascii=False))
            if review == "Check passed.":
                break
        return roles

    def generate_image_prompt_from_story(
            self,
            pages: List,
            num_turns: int = 3
        ):
        image_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_image_review_system,
                "track_history": False
            }
        })
        image_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "qwen"),
            "cfg": {
                "system_prompt": story_to_image_reviser_system,
                "track_history": False
            }
        })
        image_prompts = []

        for page in pages:
            review = ""
            image_prompt = ""
            for turn in range(num_turns):
                image_prompt, success = image_prompt_reviser.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "previous_result": image_prompt,
                    "improvement_suggestions": review,
                }, ensure_ascii=False))
                if image_prompt.startswith("Image description:"):
                    image_prompt = image_prompt[len("Image description:"):]
                review, success = image_prompt_reviewer.call(json.dumps({
                    "all_pages": pages,
                    "current_page": page,
                    "image_description": image_prompt
                }, ensure_ascii=False))
                if review == "Check passed.":
                    break
            image_prompts.append(image_prompt)
        return image_prompts


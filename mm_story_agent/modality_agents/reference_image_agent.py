import json
import torch
from PIL import Image
from typing import Dict, List, Optional
from pathlib import Path

from mm_story_agent.base import register_tool, init_tool_instance


@register_tool("reference_image_processor")
class ReferenceImageProcessor:
    """
    Agent for processing reference images for character consistency.
    Features:
    - Extract character features from reference images
    - Generate text descriptions of reference images
    - Prepare reference data for image generation
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.llm_type = cfg.get("llm", "qwen")
        self.image_size = cfg.get("image_size", (512, 512))
        self.max_references = cfg.get("max_references", 5)
        
        # Initialize LLM for image description
        self.llm_agent = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": self.get_image_description_prompt(),
                "track_history": False
            }
        })
        
        # Try to load image captioning model (optional)
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.use_captioning = True
        except ImportError:
            self.use_captioning = False
            print("BLIP not available, using LLM-based description only")

    def get_image_description_prompt(self) -> str:
        return """You are an expert at analyzing character reference images for story illustration.

Given a description of a reference image, create a detailed character description that can be used for consistent image generation.

Focus on:
1. Physical appearance (hair color, eye color, skin tone, body type, etc.)
2. Clothing style and colors
3. Facial features and expressions
4. Distinctive characteristics
5. Art style (realistic, cartoon, anime, etc.)

Respond in this JSON format:
{
    "character_description": "detailed physical description for image generation",
    "key_features": ["feature1", "feature2", "feature3"],
    "art_style": "description of artistic style",
    "clothing": "description of clothing",
    "color_palette": ["color1", "color2", "color3"]
}"""

    def load_and_process_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and preprocess reference image.
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                return None
            
            image = Image.open(image_path).convert("RGB")
            
            # Resize if needed
            if image.size != self.image_size:
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def generate_image_caption(self, image: Image.Image) -> str:
        """
        Generate caption for the image using BLIP or fallback method.
        """
        if self.use_captioning:
            try:
                inputs = self.caption_processor(image, return_tensors="pt")
                out = self.caption_model.generate(**inputs, max_length=50)
                caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
                return caption
            except Exception as e:
                print(f"Captioning failed: {e}")
                return "A character image"
        else:
            return "A character reference image"

    def extract_character_features(self, image_caption: str, character_name: str = "character") -> Dict:
        """
        Extract character features from image caption using LLM.
        """
        prompt = f"""Analyze this character image description and extract features for consistent generation:

Image description: {image_caption}
Character name: {character_name}

Create a detailed character description suitable for image generation."""
        
        try:
            response, success = self.llm_agent.call(prompt)
            if success:
                try:
                    result = json.loads(response.strip("```json").strip("```"))
                    return {
                        "success": True,
                        "character_description": result.get("character_description", ""),
                        "key_features": result.get("key_features", []),
                        "art_style": result.get("art_style", ""),
                        "clothing": result.get("clothing", ""),
                        "color_palette": result.get("color_palette", [])
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Failed to parse character feature response"
                    }
            else:
                return {
                    "success": False,
                    "error": "Character feature extraction failed"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Feature extraction error: {str(e)}"
            }

    def process_reference_images(self, reference_data: List[Dict]) -> List[Dict]:
        """
        Process multiple reference images for characters.
        
        Args:
            reference_data: List of dicts with 'image_path' and 'character_name'
        """
        processed_references = []
        
        for ref_item in reference_data[:self.max_references]:
            image_path = ref_item.get("image_path")
            character_name = ref_item.get("character_name", "character")
            
            if not image_path:
                continue
            
            # Load image
            image = self.load_and_process_image(image_path)
            if image is None:
                continue
            
            # Generate caption
            caption = self.generate_image_caption(image)
            
            # Extract features
            features = self.extract_character_features(caption, character_name)
            
            if features.get("success", False):
                processed_references.append({
                    "character_name": character_name,
                    "image_path": str(image_path),
                    "caption": caption,
                    "character_description": features["character_description"],
                    "key_features": features["key_features"],
                    "art_style": features["art_style"],
                    "clothing": features["clothing"],
                    "color_palette": features["color_palette"]
                })
        
        return processed_references

    def call(self, params: Dict) -> Dict:
        """
        Main processing method for reference images.
        
        Args:
            params: Dictionary containing:
                - reference_images: List of dicts with image_path and character_name
                - or single image_path and character_name
                
        Returns:
            Dictionary with processed reference data
        """
        # Handle single image input
        if "image_path" in params:
            reference_data = [{
                "image_path": params["image_path"],
                "character_name": params.get("character_name", "character")
            }]
        # Handle multiple images
        elif "reference_images" in params:
            reference_data = params["reference_images"]
        else:
            return {
                "success": False,
                "error": "No reference images provided"
            }
        
        # Process reference images
        processed_references = self.process_reference_images(reference_data)
        
        if processed_references:
            return {
                "success": True,
                "reference_count": len(processed_references),
                "references": processed_references
            }
        else:
            return {
                "success": False,
                "error": "No valid reference images could be processed"
            }


@register_tool("simple_reference_processor")
class SimpleReferenceProcessor:
    """
    Simpler reference image processor for basic use cases.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.image_size = cfg.get("image_size", (512, 512))

    def call(self, params: Dict) -> Dict:
        """
        Simple reference processing - just validate and prepare image paths.
        """
        # Handle single image input
        if "image_path" in params:
            image_path = Path(params["image_path"])
            character_name = params.get("character_name", "character")
            
            if not image_path.exists():
                return {
                    "success": False,
                    "error": f"Reference image not found: {image_path}"
                }
            
            return {
                "success": True,
                "reference_count": 1,
                "references": [{
                    "character_name": character_name,
                    "image_path": str(image_path),
                    "character_description": f"Character based on reference image: {character_name}",
                    "key_features": ["consistent appearance", "reference-based"],
                    "art_style": "based on reference image",
                    "clothing": "as shown in reference",
                    "color_palette": ["reference colors"]
                }]
            }
        
        # Handle multiple images
        elif "reference_images" in params:
            references = []
            for ref_item in params["reference_images"]:
                image_path = Path(ref_item.get("image_path", ""))
                character_name = ref_item.get("character_name", "character")
                
                if image_path.exists():
                    references.append({
                        "character_name": character_name,
                        "image_path": str(image_path),
                        "character_description": f"Character based on reference image: {character_name}",
                        "key_features": ["consistent appearance", "reference-based"],
                        "art_style": "based on reference image",
                        "clothing": "as shown in reference",
                        "color_palette": ["reference colors"]
                    })
            
            if references:
                return {
                    "success": True,
                    "reference_count": len(references),
                    "references": references
                }
            else:
                return {
                    "success": False,
                    "error": "No valid reference images found"
                }
        
        else:
            return {
                "success": False,
                "error": "No reference images provided"
            }
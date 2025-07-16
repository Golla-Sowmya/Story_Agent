import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from mm_story_agent.base import register_tool, init_tool_instance


@register_tool("multilingual_text_processor")
class MultilingualTextProcessor:
    """
    Agent for processing multilingual text stories.
    Features:
    - Language detection
    - Translation to English
    - Story chunking
    - Character extraction and highlighting
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.llm_type = cfg.get("llm", "qwen")
        self.chunk_size = cfg.get("chunk_size", 500)  # Target words per chunk
        self.max_chunks = cfg.get("max_chunks", 10)
        
        # Initialize LLM agent for translation and processing
        self.llm_agent = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": self.get_system_prompt(),
                "track_history": False
            }
        })
        
        # Character extraction agent
        self.character_extractor = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": self.get_character_extraction_prompt(),
                "track_history": False
            }
        })

    def get_system_prompt(self) -> str:
        return """You are a multilingual story processing assistant. Your tasks are:
1. Detect the language of the input text
2. If not in English, translate it to clear, natural English
3. Preserve the story's essence, characters, and cultural context
4. Ensure the translation is suitable for children's story generation

Always respond in this JSON format:
{
    "detected_language": "language_name",
    "original_text": "original text",
    "translated_text": "english translation",
    "confidence": "high/medium/low"
}"""

    def get_character_extraction_prompt(self) -> str:
        return """You are a character extraction specialist. Analyze the story and extract all characters with their descriptions.

Your task:
1. Identify all characters (people, animals, objects that act as characters)
2. Provide detailed descriptions for each character
3. Note their role in the story
4. Highlight key characteristics for consistent image generation

Respond in this JSON format:
{
    "characters": [
        {
            "name": "character_name",
            "description": "detailed physical and personality description",
            "role": "main/supporting/minor",
            "key_traits": ["trait1", "trait2", "trait3"]
        }
    ],
    "story_theme": "overall theme of the story",
    "setting": "where the story takes place"
}"""

    def detect_and_translate(self, text: str) -> Dict:
        """
        Detect language and translate text to English if needed.
        """
        prompt = f"Process this text and translate if needed:\n\n{text}"
        
        try:
            response, success = self.llm_agent.call(prompt)
            if success:
                # Try to parse JSON response
                try:
                    result = json.loads(response.strip("```json").strip("```"))
                    return {
                        "success": True,
                        "detected_language": result.get("detected_language", "unknown"),
                        "original_text": result.get("original_text", text),
                        "translated_text": result.get("translated_text", text),
                        "confidence": result.get("confidence", "medium")
                    }
                except json.JSONDecodeError:
                    # If JSON parsing fails, assume it's already English
                    return {
                        "success": True,
                        "detected_language": "english",
                        "original_text": text,
                        "translated_text": text,
                        "confidence": "high"
                    }
            else:
                return {
                    "success": False,
                    "error": "Translation failed"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Translation error: {str(e)}"
            }

    def extract_characters(self, text: str) -> Dict:
        """
        Extract characters from the story text.
        """
        prompt = f"Extract all characters from this story:\n\n{text}"
        
        try:
            response, success = self.character_extractor.call(prompt)
            if success:
                try:
                    result = json.loads(response.strip("```json").strip("```"))
                    return {
                        "success": True,
                        "characters": result.get("characters", []),
                        "story_theme": result.get("story_theme", ""),
                        "setting": result.get("setting", "")
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Failed to parse character extraction response"
                    }
            else:
                return {
                    "success": False,
                    "error": "Character extraction failed"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Character extraction error: {str(e)}"
            }

    def create_story_chunks(self, text: str, characters: List[Dict]) -> List[str]:
        """
        Create logical story chunks based on content and characters.
        """
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_word_count = sentence_words
            else:
                current_chunk += sentence + ". "
                current_word_count += sentence_words
            
            if len(chunks) >= self.max_chunks:
                break
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def highlight_characters_in_chunks(self, chunks: List[str], characters: List[Dict]) -> List[Dict]:
        """
        Highlight character appearances in each chunk.
        """
        highlighted_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "chunk_id": i + 1,
                "text": chunk,
                "characters_mentioned": [],
                "main_character": None
            }
            
            # Find characters mentioned in this chunk
            for character in characters:
                char_name = character["name"].lower()
                if char_name in chunk.lower():
                    chunk_info["characters_mentioned"].append({
                        "name": character["name"],
                        "description": character["description"],
                        "role": character["role"]
                    })
            
            # Determine main character for this chunk
            if chunk_info["characters_mentioned"]:
                # Prioritize main characters
                main_chars = [c for c in chunk_info["characters_mentioned"] if c["role"] == "main"]
                if main_chars:
                    chunk_info["main_character"] = main_chars[0]
                else:
                    chunk_info["main_character"] = chunk_info["characters_mentioned"][0]
            
            highlighted_chunks.append(chunk_info)
        
        return highlighted_chunks

    def call(self, params: Dict) -> Dict:
        """
        Main processing method for multilingual text stories.
        
        Args:
            params: Dictionary containing:
                - text: The story text in any language
                - output_format: "chunks" or "story_topic" (default: "chunks")
                
        Returns:
            Dictionary with processed story data
        """
        text = params.get("text", "")
        output_format = params.get("output_format", "chunks")
        
        if not text:
            return {
                "success": False,
                "error": "No text provided"
            }
        
        # Step 1: Detect language and translate if needed
        translation_result = self.detect_and_translate(text)
        if not translation_result["success"]:
            return translation_result
        
        translated_text = translation_result["translated_text"]
        
        # Step 2: Extract characters
        character_result = self.extract_characters(translated_text)
        if not character_result["success"]:
            return character_result
        
        characters = character_result["characters"]
        
        # Step 3: Create story chunks
        chunks = self.create_story_chunks(translated_text, characters)
        
        # Step 4: Highlight characters in chunks
        highlighted_chunks = self.highlight_characters_in_chunks(chunks, characters)
        
        # Format output based on request
        if output_format == "story_topic":
            # Format for story agent
            story_topic = f"Story based on: {translated_text[:200]}..."
            if translation_result["detected_language"] != "english":
                story_topic += f" (Translated from {translation_result['detected_language']})"
            
            return {
                "success": True,
                "story_topic": story_topic,
                "original_language": translation_result["detected_language"],
                "characters": characters,
                "story_theme": character_result["story_theme"],
                "setting": character_result["setting"]
            }
        else:
            # Return detailed chunk information
            return {
                "success": True,
                "original_language": translation_result["detected_language"],
                "translated_text": translated_text,
                "characters": characters,
                "story_theme": character_result["story_theme"],
                "setting": character_result["setting"],
                "chunks": highlighted_chunks,
                "chunk_count": len(highlighted_chunks)
            }


@register_tool("simple_text_translator")
class SimpleTextTranslator:
    """
    Simpler text translation agent for quick language processing.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.llm_type = cfg.get("llm", "qwen")
        
        self.translator = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": "You are a professional translator. Translate the given text to natural English while preserving the story's essence and cultural context. If the text is already in English, return it as is.",
                "track_history": False
            }
        })

    def call(self, params: Dict) -> Dict:
        """
        Simple translation method.
        """
        text = params.get("text", "")
        
        if not text:
            return {
                "success": False,
                "error": "No text provided"
            }
        
        try:
            translated_text, success = self.translator.call(f"Translate this text to English:\n\n{text}")
            
            if success:
                return {
                    "success": True,
                    "translated_text": translated_text,
                    "original_text": text
                }
            else:
                return {
                    "success": False,
                    "error": "Translation failed"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Translation error: {str(e)}"
            }
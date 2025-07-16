import json
from typing import Dict
import random

from tqdm import trange, tqdm

from ..utils.llm_output_check import parse_list
from ..base import register_tool, init_tool_instance
from ..prompts_en import question_asker_system, expert_system, \
    dlg_based_writer_system, dlg_based_writer_prompt, chapter_writer_system


def json_parse_outline(outline):
    outline = outline.strip("```json").strip("```")
    try:
        outline = json.loads(outline)
        if not isinstance(outline, dict):
            return False
        if outline.keys() == {"story_title", "story_outline"}:
            # Expected format - validate chapter structure
            for chapter in outline["story_outline"]:
                if chapter.keys() != {"chapter_title", "chapter_summary"}:
                    return False
            return True
        elif "outline" in outline:
            # Fallback format - will be normalized later
            return True
        else:
            return False
    except json.decoder.JSONDecodeError:
        return False


def normalize_outline(outline):
    """Normalize different outline formats to the expected structure."""
    if "story_title" in outline and "story_outline" in outline:
        # Already in correct format
        return outline
    
    # Handle fallback format with "outline" key
    if "outline" in outline:
        normalized = {
            "story_title": outline.get("story_title", "Untitled Story"),
            "story_outline": []
        }
        
        for i, chapter_text in enumerate(outline["outline"]):
            # Parse chapter text like "Chapter 1: Introduction - Setting up the story..."
            if " - " in chapter_text:
                title_part, summary_part = chapter_text.split(" - ", 1)
                chapter_title = title_part.strip()
                chapter_summary = summary_part.strip()
            else:
                chapter_title = chapter_text.strip()
                chapter_summary = f"Chapter {i+1} content"
            
            normalized["story_outline"].append({
                "chapter_title": chapter_title,
                "chapter_summary": chapter_summary
            })
        
        return normalized
    
    # If no recognizable format, raise error
    raise ValueError(f"Cannot normalize outline format: {outline}")


@register_tool("qa_outline_story_writer")
class QAOutlineStoryWriter:

    def __init__(self,
                 cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.max_conv_turns = cfg.get("max_conv_turns", 3)
        self.num_outline = cfg.get("num_outline", 4)
        self.llm_type = cfg.get("llm", "qwen")

    def generate_outline(self, params):
        # `params`: story setting like 
        # {
        #     "story_title": "xxx",
        #     "main_role": "xxx",
        #     ......
        # }
        asker = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": question_asker_system,
                "track_history": False
            }
        })
        expert = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": expert_system,
                "track_history": False
            }
        })

        dialogue = []
        for turn in trange(self.max_conv_turns):
            dialogue_history = "\n".join(dialogue)
            
            question, success = asker.call(
                f"Story setting: {params}\nDialogue history: \n{dialogue_history}\n",
                temperature=self.temperature
            )
            question = question.strip()
            if question == "Thank you for your help!":
                break
            dialogue.append(f"You: {question}")
            answer, success = expert.call(
                f"Story setting: {params}\nQuestion: \n{question}\nAnswer: ",
                temperature=self.temperature
            )
            answer = answer.strip()
            dialogue.append(f"Expert: {answer}")

        # print("\n".join(dialogue))
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": dlg_based_writer_system,
                "track_history": False
            }
        })
        writer_prompt = dlg_based_writer_prompt.format(
            story_setting=params,
            dialogue_history="\n".join(dialogue),
            num_outline=self.num_outline
        )

        # Try to generate outline with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                outline, success = writer.call(writer_prompt, success_check_fn=json_parse_outline)
                if success:
                    outline = json.loads(outline)
                    
                    # Normalize the outline to the expected format
                    outline = normalize_outline(outline)
                    return outline
                else:
                    print(f"Attempt {attempt + 1} failed: Invalid JSON format")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid outline after {max_retries} attempts. Last error: {e}")
                continue
        
        raise ValueError(f"Failed to generate valid outline after {max_retries} attempts")

    def generate_story_from_outline(self, outline):
        # Check if outline has the expected structure
        if not isinstance(outline, dict) or "story_outline" not in outline:
            raise ValueError(f"Invalid outline structure. Expected dict with 'story_outline' key, got: {outline}")
        
        chapter_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": chapter_writer_system,
                "track_history": False
            }
        })
        all_pages = []
        for idx, chapter in enumerate(tqdm(outline["story_outline"])):
            chapter_detail, success = chapter_writer.call(
                json.dumps(
                    {
                        "completed_story": all_pages,
                        "current_chapter": chapter
                    },
                    ensure_ascii=False
                ),
                success_check_fn=parse_list,
                temperature=self.temperature
            )
            while success is False:
                chapter_detail, success = chapter_writer.call(
                    json.dumps(
                        {
                            "completed_story": all_pages,
                            "current_chapter": chapter
                        },
                        ensure_ascii=False
                    ),
                    seed=random.randint(0, 100000),
                    temperature=self.temperature,
                    success_check_fn=parse_list
                )
            # Safe parsing: handle both list format and plain text
            try:
                # Try to parse as list first
                pages = eval(chapter_detail)
                if isinstance(pages, list):
                    pages = [page.strip() for page in pages]
                else:
                    # If not a list, treat as single page
                    pages = [str(pages).strip()]
            except (SyntaxError, ValueError):
                # If eval fails, treat as plain text (single page)
                pages = [chapter_detail.strip()]
            all_pages.extend(pages)
        # print(all_pages)
        return all_pages

    def call(self, params):
        outline = self.generate_outline(params)
        pages = self.generate_story_from_outline(outline)
        return pages

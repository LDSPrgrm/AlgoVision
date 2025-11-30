#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Configure UTF-8 encoding for stdout/stderr on Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from nicegui import app, ui, events
import asyncio
import warnings
import logging

# Suppress Windows asyncio ConnectionResetError warnings (harmless cleanup warnings)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Suppress specific asyncio warnings
    warnings.filterwarnings('ignore', category=ResourceWarning)
    logging.getLogger('asyncio').setLevel(logging.ERROR)
import textwrap
import json
import os
import re
from io import StringIO
import glob
from types import SimpleNamespace, MethodType
from pathlib import Path

# --- Imports from the original script ---
from moviepy import VideoFileClip, concatenate_videoclips
import os
import json
import uuid
import glob
import re
from dotenv import load_dotenv

# Assuming these are in a 'src' directory relative to the script
from src.config.config import Config
from utils.litellm import LiteLLMWrapper
from src.core.video_planner import VideoPlanner
from src.core.code_generator import CodeGenerator
from src.core.video_renderer import VideoRenderer
from src.utils.utils import extract_xml
from src.utils.error_recovery import ErrorRecovery
from prompts import get_banned_reasonings
from prompts.prompts_raw import (
    _code_font_size,
    _code_disable,
    _code_limit,
    _prompt_manim_cheatsheet,
)

# Load allowed models list from JSON file
allowed_models_path = os.path.join(
    os.path.dirname(__file__), "src", "utils", "models.json"
)
with open(allowed_models_path, "r", encoding="utf-8") as f:
    allowed_models = json.load(f).get("allowed_models", [])

load_dotenv(override=True)

# --- App State (Replaces st.session_state) ---
app_state = {
    "log_stream": StringIO(),
    "selected_topic": None,
    "current_topic_inspector": None,  # For the inspector tab
    "latest_pause_time": 0.0,
    "current_tcm_entry": None,
    "chat_histories": {},  # one history per topic
    "planner_model_name": next(
        (m for m in allowed_models if "gemini" in m), allowed_models[0]
    ),
    "max_retries": 5,
    "max_scene_concurrency": 5,
}


# --- All Helper Functions from the original script (remain unchanged) ---
# --- (This section is collapsed for brevity, it's identical to your script) ---
def get_topic_folders(output_dir):
    """
    Finds and returns the list of high-level topic names (e.g., "Bubble Sort").
    A valid topic folder must contain at least one run subfolder, which in turn
    must contain a scene outline.
    """
    if not os.path.exists(output_dir):
        return []
    valid_topics = set()
    for top_level_dir in os.listdir(output_dir):
        top_level_path = os.path.join(output_dir, top_level_dir)
        if os.path.isdir(top_level_path):
            # Check inside this folder for run folders (e.g., "bubble_sort")
            for inner_item in os.listdir(top_level_path):
                inner_path = os.path.join(top_level_path, inner_item)
                # If there's a sub-directory containing an outline, the top-level is a valid topic
                if os.path.isdir(inner_path) and glob.glob(
                    os.path.join(inner_path, "*_scene_outline.txt")
                ):
                    valid_topics.add(
                        top_level_dir
                    )  # Add the human-readable name, e.g., "Bubble Sort"
                    break  # Found a valid run, no need to check other runs in this topic folder
    return sorted(list(valid_topics))


def get_project_path(output_dir, topic_name):
    """
    Gets the path to the specific run folder (e.g., "output/Bubble Sort/bubble_sort").
    It finds the first sub-directory within the main topic folder.
    """
    top_level_path = os.path.join(output_dir, topic_name)
    if not os.path.isdir(top_level_path):
        # This case handles when the topic folder itself hasn't been created yet.
        return top_level_path

    # Find the first subdirectory inside the topic folder (e.g., "bubble_sort")
    for item in os.listdir(top_level_path):
        potential_path = os.path.join(top_level_path, item)
        if os.path.isdir(potential_path):
            # Return the path to the inner run folder
            return potential_path

    # Fallback if no inner run folder is found (shouldn't happen for valid topics)
    return top_level_path


def safe_read_file(path, clean=True):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not clean:
                return content
            patterns_to_remove = [
                r"</?SCENE_VISION_STORYBOARD_PLAN>",
                r"</?SCENE_TECHNICAL_IMPLEMENTATION_PLAN>",
                r"</?SCENE_ANIMATION_NARRATION_PLAN>",
                r"# Scene \d+ Implementation Plan",
                r"\[SCENE_VISION\]",
                r"\[STORYBOARD\]",
                r"\[ANIMATION_STRATEGY\]",
                r"\[NARRATION\]",
                r"\[ANIMATION:.*?\]",
            ]
            for pattern in patterns_to_remove:
                content = re.sub(pattern, "", content)
            return content.strip()
    except FileNotFoundError:
        return f"File not found at: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def check_status(self, theorem: dict):
    topic = theorem["theorem"]
    project_path = get_project_path(self.output_dir, topic)
    inner_folder_name = os.path.basename(project_path)
    file_prefix = inner_folder_name
    scene_outline_path = os.path.join(project_path, f"{file_prefix}_scene_outline.txt")
    has_scene_outline = os.path.exists(scene_outline_path)
    num_scenes = 0
    if has_scene_outline:
        with open(scene_outline_path, "r") as f:
            scene_outline = f.read()
            scene_outline_content = extract_xml(scene_outline, "SCENE OUTLINE")
            num_scenes = len(re.findall(r"<SCENE_(\d+)>[^<]", scene_outline_content))
    implementation_plans, code_files, rendered_scenes = 0, 0, 0
    scene_status = []
    for i in range(1, num_scenes + 1):
        scene_dir = os.path.join(project_path, f"scene{i}")
        plan_path = os.path.join(
            scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt"
        )
        has_plan = os.path.exists(plan_path)
        if has_plan:
            implementation_plans += 1
        code_dir = os.path.join(scene_dir, "code")
        has_code = os.path.exists(code_dir) and any(
            f.endswith(".py") for f in os.listdir(code_dir)
        )
        if has_code:
            code_files += 1
        has_render = os.path.exists(os.path.join(scene_dir, "succ_rendered.txt"))
        if has_render:
            rendered_scenes += 1
        scene_status.append(
            {
                "scene_number": i,
                "has_plan": has_plan,
                "has_code": has_code,
                "has_render": has_render,
            }
        )
    has_combined_video = os.path.exists(
        os.path.join(project_path, f"{file_prefix}_combined.mp4")
    )
    return {
        "topic": topic,
        "has_scene_outline": has_scene_outline,
        "total_scenes": num_scenes,
        "implementation_plans": implementation_plans,
        "code_files": code_files,
        "rendered_scenes": rendered_scenes,
        "has_combined_video": has_combined_video,
        "scene_status": scene_status,
    }


def set_active_output_dir(self, new_output_dir):
    self.output_dir = new_output_dir
    self.planner.output_dir = new_output_dir
    self.code_generator.output_dir = new_output_dir
    self.video_renderer.output_dir = new_output_dir


def load_voices(file_path="voices.json"):
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            voices = json.load(f)
            return [v for v in voices if "id" in v and "name" in v]
    except Exception as e:
        ui.notify(f"Error loading voices.json: {e}", type="negative")
        return []


# --- UPDATED/NEW HELPER FUNCTIONS ---


def find_latest_video_for_scene(project_path: str, scene_num: int) -> str | None:
    """
    Finds the latest rendered video for a given scene number in the project.
    (Implementation from Streamlit script)
    """
    videos_dir = os.path.join(project_path, "media", "videos")
    search_pattern = os.path.join(videos_dir, f"*scene{scene_num}_v*")
    potential_folders = glob.glob(search_pattern)
    if not potential_folders:
        print(f"  - No video folders found for scene {scene_num}.")
        return None

    def get_version(path):
        m = re.search(r"_v(\d+)", path)
        return int(m.group(1)) if m else -1

    latest_folder = max(potential_folders, key=get_version)
    for res in ["1080p60", "720p30", "480p15"]:
        video_file = os.path.join(latest_folder, res, f"Scene{scene_num}.mp4")
        if os.path.exists(video_file):
            return video_file
    print(f"  - No video file found for scene {scene_num} in {latest_folder}.")
    return None


def split_narration_to_chunks(narration, max_chars=40, max_lines=2):
    """
    Split narration into chunks suitable for SRT (by sentences, then by line length).
    (New function from Streamlit script)
    """
    sentences = re.split(r"(?<=[.!?])\s+", narration)
    chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        wrapped = textwrap.wrap(sentence.strip(), width=max_chars)
        for i in range(0, len(wrapped), max_lines):
            block = wrapped[i : i + max_lines]
            chunks.append(" ".join(block))
    return chunks


def tcm_to_srt(
    tcm: list, max_line_length=40, max_lines=2, max_block_duration=4.0
) -> str:
    """
    Convert TCM events to SRT, splitting long narration and duration into multiple blocks.
    (Implementation from Streamlit script)
    """

    def sec_to_srt_time(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:003}"

    srt_lines = []
    idx = 1
    for event in tcm:
        narration = event.get("narrationText", "").strip()
        if not narration or narration == "...":
            continue
        start = float(event["startTime"])
        end = float(event["endTime"])
        duration = end - start
        chunks = split_narration_to_chunks(
            narration, max_chars=max_line_length, max_lines=max_lines
        )
        n_chunks = len(chunks)
        if n_chunks == 0:
            continue
        chunk_duration = min(duration / n_chunks, max_block_duration)
        chunk_start = start
        for chunk in chunks:
            chunk_end = min(chunk_start + chunk_duration, end)
            srt_lines.append(f"{idx}")
            srt_lines.append(
                f"{sec_to_srt_time(chunk_start)} --> {sec_to_srt_time(chunk_end)}"
            )
            wrapped = textwrap.wrap(chunk, width=max_line_length)
            if len(wrapped) > max_lines:
                wrapped = wrapped[: max_lines - 1] + [
                    " ".join(wrapped[max_lines - 1 :])
                ]
            srt_lines.extend(wrapped)
            srt_lines.append("")
            idx += 1
            chunk_start = chunk_end
            if chunk_start >= end:
                break
    return "\n".join(srt_lines)


def srt_to_vtt(srt_content: str) -> str:
    """Convert SRT subtitle format to WebVTT format."""
    lines = srt_content.strip().split('\n')
    vtt_lines = ['WEBVTT\n']
    
    for line in lines:
        # Convert SRT timestamp format (00:00:00,000) to VTT format (00:00:00.000)
        if '-->' in line:
            line = line.replace(',', '.')
        vtt_lines.append(line)
    
    return '\n'.join(vtt_lines)


def combine_videos(topic: str, output_dir: str = "output"):
    """
    Combines all videos for a topic and generates the final, fine-grained TCM and SRT subtitles.
    (Implementation from Streamlit script, adapted with NiceGUI notifications)
    """
    project_path = get_project_path(output_dir, topic)
    project_name = os.path.basename(os.path.dirname(project_path))
    inner_folder_name = os.path.basename(project_path)

    output_video_path = os.path.join(project_path, f"{inner_folder_name}_combined.mp4")
    output_tcm_path = os.path.join(
        project_path, f"{inner_folder_name}_combined_tcm.json"
    )
    output_srt_path = os.path.join(project_path, f"{inner_folder_name}_combined.srt")
    output_vtt_path = os.path.join(project_path, f"{inner_folder_name}_combined.vtt")

    if (
        os.path.exists(output_video_path)
        and os.path.exists(output_tcm_path)
        and os.path.exists(output_srt_path)
        and os.path.exists(output_vtt_path)
    ):
        msg = f"[{topic}] Combined assets already exist. Skipping."
        print(msg)
        return "already_exists"

    print(
        f"[{topic}] ==> Finalizing project: Combining videos and creating global TCM..."
    )
    final_tcm, video_clips_paths, global_time_offset = [], [], 0.0

    try:
        scene_dirs = sorted(
            glob.glob(os.path.join(project_path, "scene*")),
            key=lambda x: int(re.search(r"scene(\d+)", x).group(1)),
        )
    except (TypeError, ValueError):
        print(
            f"  - ERROR: Could not sort scene directories in '{project_path}'. Check folder names."
        )
        return

    for scene_dir in scene_dirs:
        scene_num = int(re.search(r"scene(\d+)", os.path.basename(scene_dir)).group(1))
        video_path = find_latest_video_for_scene(project_path, scene_num)
        proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")
        succ_rendered_path = os.path.join(scene_dir, "succ_rendered.txt")

        if not video_path or not os.path.exists(succ_rendered_path):
            print(
                f"  - Skipping Scene {scene_num}: Video or succ_rendered.txt missing."
            )
            continue

        if not os.path.exists(proto_tcm_path):
            print(
                f"  - WARNING: proto_tcm.json not found for Scene {scene_num}. Skipping fine-grained analysis."
            )
            continue

        try:
            with VideoFileClip(video_path) as clip:
                actual_duration = clip.duration
            video_clips_paths.append(video_path)

            with open(proto_tcm_path, "r", encoding="utf-8") as f:
                proto_tcm = json.load(f)

            total_estimated_duration = sum(
                e.get("estimatedDuration", 1.0) for e in proto_tcm
            )
            scaling_factor = (
                actual_duration / total_estimated_duration
                if total_estimated_duration > 0
                else 1
            )

            scene_time_offset = 0
            for event in proto_tcm:
                scaled_duration = event.get("estimatedDuration", 1.0) * scaling_factor
                event["startTime"] = f"{global_time_offset + scene_time_offset:.3f}"
                event["endTime"] = (
                    f"{global_time_offset + scene_time_offset + scaled_duration:.3f}"
                )
                event["conceptId"] = (
                    f"{project_name}.scene_{scene_num}.{event.get('conceptName', 'event').replace(' ', '_')}"
                )
                if "estimatedDuration" in event:
                    del event["estimatedDuration"]
                final_tcm.append(event)
                scene_time_offset += scaled_duration

            print(
                f"  - Processed Scene {scene_num} (Duration: {actual_duration:.2f}s), {len(proto_tcm)} TCM entries."
            )
            global_time_offset += actual_duration
        except Exception as e:
            print(f"  - ERROR processing Scene {scene_num}: {e}")

    if video_clips_paths:
        clips = [VideoFileClip(p) for p in video_clips_paths]
        final_video_clip = concatenate_videoclips(clips)

        final_video_clip.write_videofile(
            output_video_path, codec="libx264", audio_codec="aac", logger="bar"
        )
        with open(output_tcm_path, "w", encoding="utf-8") as f:
            json.dump(final_tcm, f, indent=2, ensure_ascii=False)

        srt_content = tcm_to_srt(final_tcm)
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        # Also create VTT version for better browser compatibility
        vtt_content = srt_to_vtt(srt_content)
        with open(output_vtt_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)

        print(f"[{topic}] ==> Project finalized.")
        for clip in clips:
            clip.close()
        final_video_clip.close()
        return "success"
    else:
        print(f"[{topic}] <== No rendered scenes found to finalize.")
        return "no_scenes"


# --- VideoGenerator Class (from generate_video.py) ---
class VideoGenerator:
    """
    A class for generating manim videos using AI models.

    This class coordinates the video generation pipeline by managing scene planning,
    code generation, and video rendering. It supports concurrent scene processing,
    visual code fixing, and RAG (Retrieval Augmented Generation).
    """

    def __init__(
        self,
        planner_model,
        scene_model=None,
        output_dir="output",
        verbose=False,
        use_rag=False,
        use_context_learning=False,
        context_learning_path="data/context_learning",
        chroma_db_path="data/rag/chroma_db",
        manim_docs_path="data/rag/manim_docs",
        embedding_model="azure/text-embedding-3-large",
        use_visual_fix_code=False,
        use_langfuse=True,
        trace_id=None,
        max_scene_concurrency: int = 5,
    ):
        print("Initializing VideoGenerator...")
        self.output_dir = output_dir
        self.verbose = verbose
        self.use_visual_fix_code = use_visual_fix_code
        self.session_id = self._load_or_create_session_id()
        self.scene_semaphore = asyncio.Semaphore(max_scene_concurrency)
        print(f"Scene concurrency limit set to: {max_scene_concurrency}")
        self.banned_reasonings = get_banned_reasonings()
        self.failed_scenes = []
        self.error_recovery = ErrorRecovery(output_dir)
        self.rate_limit_detected = False
        self.last_rate_limit_time = 0

        print("Initializing sub-modules: VideoPlanner, CodeGenerator, VideoRenderer...")
        self.planner = VideoPlanner(
            planner_model=planner_model,
            output_dir=output_dir,
            print_response=verbose,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            use_rag=use_rag,
            session_id=self.session_id,
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            use_langfuse=use_langfuse,
        )
        self.code_generator = CodeGenerator(
            scene_model=scene_model if scene_model is not None else planner_model,
            output_dir=output_dir,
            print_response=verbose,
            use_rag=use_rag,
            use_context_learning=use_context_learning,
            context_learning_path=context_learning_path,
            chroma_db_path=chroma_db_path,
            manim_docs_path=manim_docs_path,
            embedding_model=embedding_model,
            use_visual_fix_code=use_visual_fix_code,
            use_langfuse=use_langfuse,
            session_id=self.session_id,
        )
        self.video_renderer = VideoRenderer(
            output_dir=output_dir,
            print_response=verbose,
            use_visual_fix_code=self.use_visual_fix_code,
        )
        print("VideoGenerator initialized successfully.")

    def _load_or_create_session_id(self) -> str:
        """Load existing session ID from file or create a new one."""
        session_file = os.path.join(self.output_dir, "session_id.txt")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_id = f.read().strip()
                return session_id
        session_id = str(uuid.uuid4())
        print(f"No existing session ID found. Creating a new one: {session_id}")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(session_file, "w", encoding='utf-8') as f:
            f.write(session_id)
        print(f"Saved new session ID to {session_file}")
        return session_id

    def generate_scene_outline(self, topic: str, description: str, session_id: str) -> str:
        """Generate scene outline using VideoPlanner."""
        print(f"[{topic}] ==> Generating scene outline...")
        outline = self.planner.generate_scene_outline(topic, description, session_id)
        print(f"[{topic}] ==> Scene outline generated successfully.")
        return outline

    async def generate_scene_implementation_concurrently(
        self, topic: str, description: str, plan: str, session_id: str
    ):
        """Generate scene implementations concurrently using VideoPlanner."""
        print(f"[{topic}] ==> Generating scene implementations concurrently...")
        implementations = await self.planner.generate_scene_implementation_concurrently(
            topic, description, plan, session_id, self.scene_semaphore
        )
        print(f"[{topic}] ==> All concurrent scene implementations generated.")
        return implementations

    async def render_video_fix_code(
        self,
        topic: str,
        description: str,
        scene_outline: str,
        implementation_plans: list,
        max_retries=3,
        session_id: str = None,
    ):
        """Render the video for all scenes with code fixing capability."""
        print(f"[{topic}] ==> Preparing to render {len(implementation_plans)} scenes...")
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)

        tasks = []
        for scene_num, implementation_plan in implementation_plans:
            scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{scene_num}")
            subplan_dir = os.path.join(scene_dir, "subplans")
            os.makedirs(subplan_dir, exist_ok=True)

            scene_trace_id_path = os.path.join(subplan_dir, "scene_trace_id.txt")
            try:
                with open(scene_trace_id_path, "r") as f:
                    scene_trace_id = f.read().strip()
            except FileNotFoundError:
                scene_trace_id = str(uuid.uuid4())
                with open(scene_trace_id_path, "w", encoding='utf-8') as f:
                    f.write(scene_trace_id)

            proto_tcm_str = ""
            proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")
            if os.path.exists(proto_tcm_path):
                with open(proto_tcm_path, "r") as f:
                    proto_tcm_str = f.read()

            task = self.process_scene(
                scene_num - 1,
                scene_outline,
                implementation_plan,
                proto_tcm_str,
                topic,
                description,
                max_retries,
                file_prefix,
                session_id,
                scene_trace_id,
            )
            tasks.append(task)

        print(f"[{topic}] Starting concurrent processing of {len(tasks)} scenes...")
        await asyncio.gather(*tasks)
        print(f"[{topic}] <== All scene processing tasks completed.")

    def _save_topic_session_id(self, topic: str, session_id: str):
        """Save session ID for a specific topic."""
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        topic_dir = os.path.join(self.output_dir, topic, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)
        session_file = os.path.join(topic_dir, "session_id.txt")
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(session_id)

    def _load_topic_session_id(self, topic: str):
        """Load session ID for a specific topic if it exists."""
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)
        session_file = os.path.join(self.output_dir, topic, file_prefix, "session_id.txt")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                return f.read().strip()
        return None

    def cleanup_invalid_success_markers(self, topic: str) -> int:
        """Remove succ_rendered.txt files for scenes that don't actually have rendered videos."""
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        topic_dir = os.path.join(self.output_dir, topic, file_prefix)
        if not os.path.exists(topic_dir):
            return 0
        removed_count = 0
        scene_dirs = glob.glob(os.path.join(topic_dir, "scene*"))
        for scene_dir in scene_dirs:
            if not os.path.isdir(scene_dir):
                continue
            scene_match = re.search(r'scene(\d+)', os.path.basename(scene_dir))
            if not scene_match:
                continue
            scene_num = scene_match.group(1)
            succ_file = os.path.join(scene_dir, "succ_rendered.txt")
            if os.path.exists(succ_file):
                media_dir = os.path.join(topic_dir, "media", "videos")
                video_pattern = os.path.join(media_dir, f"{file_prefix}_scene{scene_num}_v*")
                video_folders = glob.glob(video_pattern)
                has_video = False
                for video_folder in video_folders:
                    for res_dir in ["1080p60", "720p30", "480p15"]:
                        video_file = os.path.join(video_folder, res_dir, f"Scene{scene_num}.mp4")
                        if os.path.exists(video_file):
                            has_video = True
                            break
                    if has_video:
                        break
                if not has_video:
                    os.remove(succ_file)
                    removed_count += 1
        return removed_count

    def load_implementation_plans(self, topic: str):
        """Load implementation plans for each scene."""
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)
        scene_outline_path = os.path.join(
            self.output_dir, topic, file_prefix, f"{file_prefix}_scene_outline.txt"
        )
        if not os.path.exists(scene_outline_path):
            return {}
        with open(scene_outline_path, "r") as f:
            scene_outline = f.read()
        scene_outline_content = extract_xml(scene_outline, "SCENE_OUTLINE")
        scene_number = 0
        if scene_outline_content:
            scene_number = len(re.findall(r"<SCENE_(\d+)>[^<]", scene_outline_content))
        implementation_plans = {}
        for i in range(1, scene_number + 1):
            plan_path = os.path.join(
                self.output_dir, topic, file_prefix, f"scene{i}",
                f"{file_prefix}_scene{i}_implementation_plan.txt",
            )
            if os.path.exists(plan_path):
                with open(plan_path, "r") as f:
                    implementation_plans[i] = f.read()
            else:
                implementation_plans[i] = None
        return implementation_plans

    async def _generate_scene_implementation_single(
        self, topic: str, description: str, scene_outline_i: str, i: int,
        file_prefix: str, session_id: str, scene_trace_id: str
    ):
        """Orchestrates the generation of a detailed plan and Proto-TCM for a single scene."""
        full_llm_response_obj = await self.planner._generate_scene_implementation_single(
            topic, description, scene_outline_i, i, file_prefix, session_id, scene_trace_id
        )
        if isinstance(full_llm_response_obj, dict) and "plan" in full_llm_response_obj and "proto_tcm" in full_llm_response_obj:
            plan = full_llm_response_obj["plan"]
            proto_tcm_str = full_llm_response_obj["proto_tcm"]
        else:
            full_llm_response = ""
            if isinstance(full_llm_response_obj, str):
                full_llm_response = full_llm_response_obj
            elif isinstance(full_llm_response_obj, dict):
                try:
                    full_llm_response = full_llm_response_obj["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    if "content" in full_llm_response_obj:
                        full_llm_response = full_llm_response_obj["content"]
                    else:
                        full_llm_response = str(full_llm_response_obj)
            else:
                full_llm_response = str(full_llm_response_obj)
            plan = extract_xml(full_llm_response, "SCENE_TECHNICAL_IMPLEMENTATION_PLAN")
            if not plan or "<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>" not in plan:
                plan = full_llm_response
            proto_tcm_str = extract_xml(full_llm_response, "SCENE_PROTO_TCM")
        scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{i}")
        os.makedirs(scene_dir, exist_ok=True)
        if proto_tcm_str and "<SCENE_PROTO_TCM>" not in proto_tcm_str:
            try:
                proto_tcm_data = json.loads(proto_tcm_str)
                proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")
                with open(proto_tcm_path, "w", encoding="utf-8") as f:
                    json.dump(proto_tcm_data, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                proto_tcm_str = ""
        else:
            proto_tcm_str = ""
        plan_path = os.path.join(scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt")
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan)
        return {"plan": plan, "proto_tcm": proto_tcm_str}

    async def generate_video_pipeline(
        self, topic: str, description: str, max_retries: int,
        only_plan: bool = False, specific_scenes: list = None, progress_callback=None
    ):
        """Modified pipeline to handle partial scene completions."""
        # Create a simple args object for compatibility
        class Args:
            only_render = False
        args = Args()
        
        topic_folder_session_file = os.path.join(self.output_dir, topic, "session_id.txt")
        if os.path.exists(topic_folder_session_file):
            with open(topic_folder_session_file, "r") as f:
                session_id = f.read().strip()
        else:
            session_id = self._load_or_create_session_id()
            os.makedirs(os.path.join(self.output_dir, topic), exist_ok=True)
            with open(topic_folder_session_file, "w", encoding='utf-8') as f:
                f.write(session_id)
        self._save_topic_session_id(topic, session_id)
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)
        scene_outline_path = os.path.join(
            self.output_dir, topic, file_prefix, f"{file_prefix}_scene_outline.txt"
        )
        if progress_callback:
            progress_callback(0.05, "📝 Planning your video structure...")
        if os.path.exists(scene_outline_path):
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
        else:
            scene_outline_obj = self.planner.generate_scene_outline(topic, description, session_id)
            scene_outline = ""
            if isinstance(scene_outline_obj, str):
                scene_outline = scene_outline_obj
            elif isinstance(scene_outline_obj, dict):
                try:
                    scene_outline = scene_outline_obj["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    if "content" in scene_outline_obj:
                        scene_outline = scene_outline_obj["content"]
                    else:
                        scene_outline = str(scene_outline_obj)
            else:
                scene_outline = str(scene_outline_obj)
            if not scene_outline or "<SCENE_OUTLINE>" not in scene_outline:
                raise ValueError(f"[{topic}] FAILED to generate a valid scene outline.")
            os.makedirs(os.path.join(self.output_dir, topic, file_prefix), exist_ok=True)
            with open(scene_outline_path, "w", encoding="utf-8") as f:
                f.write(scene_outline)
        if progress_callback:
            progress_callback(0.15, "✅ Video structure ready")
        removed = self.cleanup_invalid_success_markers(topic)
        if progress_callback:
            progress_callback(0.20, "🎨 Designing each scene...")
        implementation_plans_dict = self.load_implementation_plans(topic)
        scene_outline_content = extract_xml(scene_outline, "SCENE_OUTLINE")
        scene_numbers = len(re.findall(r"<SCENE_(\d+)>[^<]", scene_outline_content)) if scene_outline_content else 0
        missing_scenes = []
        for i in range(1, scene_numbers + 1):
            if implementation_plans_dict.get(i) is None and (specific_scenes is None or i in specific_scenes):
                missing_scenes.append(i)
        if missing_scenes:
            for idx, scene_num in enumerate(missing_scenes):
                if progress_callback:
                    plan_progress = 0.20 + (0.15 * (idx / len(missing_scenes)))
                    progress_callback(plan_progress, f"🎨 Designing scene {scene_num} of {scene_numbers}...")
                scene_match = re.search(f"<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>", scene_outline_content, re.DOTALL)
                if scene_match:
                    scene_outline_i = scene_match.group(1)
                    scene_trace_id = str(uuid.uuid4())
                    implementation_details = await self._generate_scene_implementation_single(
                        topic, description, scene_outline_i, scene_num, file_prefix, session_id, scene_trace_id
                    )
                    implementation_plans_dict[scene_num] = implementation_details["plan"]
        if progress_callback:
            progress_callback(0.35, "✅ All scenes designed")
        if only_plan:
            return
        if progress_callback:
            progress_callback(0.40, "🎬 Creating animations...")
        sorted_scene_numbers = sorted(implementation_plans_dict.keys())
        processed_count = 0
        total_scenes_to_process = len([s for s in sorted_scene_numbers if not specific_scenes or s in specific_scenes])
        for scene_num in sorted_scene_numbers:
            if specific_scenes and scene_num not in specific_scenes:
                continue
            scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{scene_num}")
            is_rendered = os.path.exists(os.path.join(scene_dir, "succ_rendered.txt"))
            if is_rendered and not args.only_render:
                continue
            implementation_plan = implementation_plans_dict.get(scene_num)
            if not implementation_plan:
                continue
            proto_tcm_str = ""
            proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")
            if os.path.exists(proto_tcm_path):
                with open(proto_tcm_path, "r") as f:
                    proto_tcm_str = f.read()
            scene_trace_id_path = os.path.join(scene_dir, "subplans", "scene_trace_id.txt")
            if os.path.exists(scene_trace_id_path):
                with open(scene_trace_id_path, "r") as f:
                    scene_trace_id = f.read().strip()
            else:
                os.makedirs(os.path.dirname(scene_trace_id_path), exist_ok=True)
                scene_trace_id = str(uuid.uuid4())
                with open(scene_trace_id_path, "w", encoding='utf-8') as f:
                    f.write(scene_trace_id)
            if progress_callback:
                scene_progress = 0.40 + (0.50 * (processed_count / total_scenes_to_process))
                progress_callback(scene_progress, f"🎬 Animating scene {scene_num} of {scene_numbers}...")
            await self.process_scene(
                i=scene_num - 1, scene_outline=scene_outline, scene_implementation=implementation_plan,
                proto_tcm=proto_tcm_str, topic=topic, description=description, max_retries=max_retries,
                file_prefix=file_prefix, session_id=session_id, scene_trace_id=scene_trace_id,
            )
            processed_count += 1
            if progress_callback:
                scene_progress = 0.40 + (0.50 * (processed_count / total_scenes_to_process))
                progress_callback(scene_progress, f"✅ Scene {scene_num} done!")
        if progress_callback:
            progress_callback(0.90, "✅ All animations complete!")

    async def process_scene(
        self, i: int, scene_outline: str, scene_implementation: str, proto_tcm: str,
        topic: str, description: str, max_retries: int, file_prefix: str,
        session_id: str, scene_trace_id: str,
    ):
        """Process a single scene using CodeGenerator and VideoRenderer."""
        curr_scene = i + 1
        curr_version = 0
        rag_queries_cache = {}
        code_dir = os.path.join(
            self.output_dir, topic, file_prefix, f"scene{curr_scene}", "code"
        )
        os.makedirs(code_dir, exist_ok=True)
        media_dir = os.path.join(self.output_dir, topic, file_prefix, "media")
        async with self.scene_semaphore:
            print(f"Scene {curr_scene} ---> Generating animation code...")
            code, log = self.code_generator.generate_manim_code(
                topic=topic, description=description, scene_outline=scene_outline,
                scene_implementation=scene_implementation, proto_tcm=proto_tcm,
                scene_number=curr_scene, additional_context=[
                    _prompt_manim_cheatsheet, _code_font_size, _code_limit, _code_disable,
                ], scene_trace_id=scene_trace_id, session_id=session_id,
                rag_queries_cache=rag_queries_cache,
            )
            log_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_init_log.txt")
            code_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Scene {curr_scene} ✅ Animation code generated.")
            print(f"Scene {curr_scene} ---> Rendering animation...")
            error_message = None
            while curr_version < max_retries:
                code, error_message = await self.video_renderer.render_scene(
                    code=code, file_prefix=file_prefix, curr_scene=curr_scene,
                    curr_version=curr_version, code_dir=code_dir, media_dir=media_dir,
                    max_retries=max_retries, use_visual_fix_code=self.use_visual_fix_code,
                    visual_self_reflection_func=self.code_generator.visual_self_reflection,
                    banned_reasonings=self.banned_reasonings, scene_trace_id=scene_trace_id,
                    topic=topic, session_id=session_id,
                )
                if error_message is None:
                    print(f"Scene {curr_scene} ✅ Animation rendered successfully.")
                    break
                curr_version += 1
                if curr_version >= max_retries:
                    print(f"Scene {curr_scene} ⚠️ Failed to render after {max_retries} attempts. Check logs for details.")
                    self.failed_scenes.append({
                        'topic': topic, 'scene': curr_scene,
                        'last_error': error_message, 'total_attempts': curr_version + 1,
                    })
                    break
                print(f"Scene {curr_scene} ---> Fixing code issues (attempt {curr_version + 1}/{max_retries})...")
                print(f"Scene {curr_scene} ---> Analyzing error and generating fix...")
                code, log = self.code_generator.fix_code_errors(
                    implementation_plan=scene_implementation, proto_tcm=proto_tcm,
                    code=code, error=error_message, scene_trace_id=scene_trace_id,
                    topic=topic, scene_number=curr_scene, session_id=session_id,
                    rag_queries_cache=rag_queries_cache,
                )
                print(f"Scene {curr_scene} ---> Fixed code generated, saving...")
                log_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_fix_log.txt")
                code_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(log)
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(code)
                print(f"Scene {curr_scene} ---> Re-rendering with fixed code...")


# --- Initialize Video Generator ---
def get_video_generator(
    planner_name,
    scene_concurrency,
):
    llm_kwargs = {
        "temperature": 0.7,
        "print_cost": True,
    }
    planner_model, scene_model = LiteLLMWrapper(
        model_name=planner_name, **llm_kwargs
    ), LiteLLMWrapper(model_name=planner_name, **llm_kwargs)
    return VideoGenerator(
        planner_model=planner_model,
        scene_model=scene_model,
        output_dir="output",
        max_scene_concurrency=scene_concurrency,
    )


try:
    video_generator = get_video_generator(
        app_state["planner_model_name"], app_state["max_scene_concurrency"]
    )
    video_generator.check_theorem_status = MethodType(
        check_status, video_generator
    )
    video_generator.set_active_output_dir = MethodType(
        set_active_output_dir, video_generator
    )
except Exception as e:
    print(f"Failed to initialize VideoGenerator: {e}")
    video_generator = None

# --- UI Definition ---
app.add_static_files("/output", "output")

# --- Streamlit-Inspired Color Scheme ---
THEME_COLORS = {
    "primary": "#FF4B4B",  # Streamlit's signature red
    "secondary": "#4A4A4A",  # Dark gray for text and secondary elements
    "accent": "#00A2FF",  # A bright blue for accents
    "positive": "#28A745",  # Success green
    "negative": "#DC3545",  # Error red
    "info": "#17A2B8",  # Informational teal
    "warning": "#FFC107",  # Warning yellow
}


@ui.page("/")
async def main_page():
    # --- Page Configuration ---
    ui.colors(
        primary=THEME_COLORS["primary"],
        secondary=THEME_COLORS["secondary"],
        accent=THEME_COLORS["accent"],
        positive=THEME_COLORS["positive"],
        negative=THEME_COLORS["negative"],
        info=THEME_COLORS["info"],
        warning=THEME_COLORS["warning"],
    )
    ui.add_head_html(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
    )
    ui.add_body_html('''
        <style>
            /* Global Streamlit-inspired styling */
            :root {
                --bg-primary: #fafafa;
                --bg-card: #ffffff;
                --text-primary: #1f2937;
                --text-secondary: #6b7280;
                --border-color: #e6e9ef;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
                background-color: var(--bg-primary) !important;
                color: var(--text-primary) !important;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            /* Dark mode variables - works with any dark class */
            .dark,
            body.dark,
            html.dark,
            body.body--dark {
                --bg-primary: #0e1117;
                --bg-card: #1a1d24;
                --text-primary: #e5e7eb;
                --text-secondary: #9ca3af;
                --border-color: #262c36;
            }
            
            /* Apply dark mode to body when any dark class is present */
            .dark body,
            body.dark,
            html.dark body,
            body.body--dark {
                background-color: #0e1117 !important;
                color: #e5e7eb !important;
            }
            
            /* Force all text elements to use theme colors in dark mode */
            .dark *:not(.q-btn__content):not([class*="text-white"]):not([class*="text-primary"]),
            body.dark *:not(.q-btn__content):not([class*="text-white"]):not([class*="text-primary"]),
            html.dark *:not(.q-btn__content):not([class*="text-white"]):not([class*="text-primary"]),
            body.body--dark *:not(.q-btn__content):not([class*="text-white"]):not([class*="text-primary"]) {
                color: #e5e7eb !important;
            }
            
            body.dark,
            .dark body,
            body.body--dark {
                background-color: #0e1117 !important;
            }
            
            /* Sidebar edge indicator when collapsed */
            .q-drawer--closed + .q-page-container::before {
                content: '';
                position: fixed;
                left: 0;
                top: 64px;
                bottom: 0;
                width: 4px;
                background: linear-gradient(180deg, 
                    transparent 0%, 
                    var(--q-primary) 50%, 
                    transparent 100%);
                opacity: 0.3;
                z-index: 1000;
                pointer-events: none;
                transition: opacity 0.3s ease;
            }
            
            .q-drawer--closed + .q-page-container:hover::before {
                opacity: 0.6;
            }
            
            /* FORCE DISABLE all transitions and animations in drawer content */
            .q-drawer * {
                transition: none !important;
                animation: none !important;
            }
            
            /* Dark mode support - use multiple selectors for compatibility */
            .dark,
            body.dark,
            body.body--dark,
            html.dark body {
                --dark-bg: #0e1117;
                --dark-card: #1a1d24;
                --dark-border: #262c36;
                --dark-text: #e5e7eb;
                --dark-text-secondary: #9ca3af;
            }
            
            /* But keep the drawer itself smooth */
            .q-drawer {
                transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            /* Scroll indicator with primary color */
            .custom-scrollbar::after {
                content: '';
                position: sticky;
                bottom: 0;
                left: 0;
                right: 0;
                height: 20px;
                background: linear-gradient(to top, rgba(255, 75, 75, 0.1), transparent);
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .custom-scrollbar:not(:hover)::after {
                opacity: 0;
            }
            
            /* Separator with primary accent - minimal spacing */
            .q-drawer .q-separator {
                background: linear-gradient(90deg, transparent 0%, #FF4B4B 50%, transparent 100%) !important;
                opacity: 0.2 !important;
                margin: 0 !important;
                height: 1px !important;
            }
            
            /* Remove default separator margins in expansion items */
            .q-expansion-item .q-separator {
                margin: 0 !important;
            }
            
            /* Enhanced Card styling with glass morphism */
            .q-card {
                border-radius: 12px !important;
                border: 1px solid #e6e9ef !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
                background-color: white !important;
                backdrop-filter: blur(10px) !important;
                transition: all 0.3s ease !important;
            }
            
            .q-card:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
            }
            
            /* Dark mode cards - multiple selector support */
            body.dark .q-card,
            .dark .q-card,
            body.body--dark .q-card,
            html.dark .q-card {
                border-color: #262c36 !important;
                background-color: rgba(26, 29, 36, 0.95) !important;
            }
            
            body.dark .q-card:hover,
            .dark .q-card:hover,
            body.body--dark .q-card:hover,
            html.dark .q-card:hover {
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
                border-color: #374151 !important;
            }
            
            /* Glass effect for sidebar sections */
            .q-drawer .q-card,
            .q-drawer .q-expansion-item {
                backdrop-filter: blur(8px) saturate(180%) !important;
            }
            
            /* Enhanced Button styling with modern effects */
            .q-btn {
                border-radius: 10px !important;
                text-transform: none !important;
                font-weight: 500 !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
                letter-spacing: 0.01em !important;
            }
            
            .q-btn:not(.q-btn--flat):hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
            }
            
            .q-btn--flat:hover {
                background-color: rgba(0, 0, 0, 0.05) !important;
            }
            
            body.dark .q-btn--flat:hover {
                background-color: rgba(255, 255, 255, 0.08) !important;
            }
            
            .q-btn:active {
                transform: translateY(0) scale(0.98);
            }
            
            .q-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none !important;
            }
            
            /* Round buttons (icon buttons) */
            .q-btn--round {
                border-radius: 50% !important;
            }
            
            .q-btn--round:hover {
                transform: scale(1.1) !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12) !important;
            }
            
            /* Primary button glow effect */
            .q-btn--unelevated.bg-primary:hover {
                box-shadow: 0 0 20px rgba(255, 75, 75, 0.4) !important;
            }
            
            /* Enhanced Input fields with primary focus */
            .q-field {
                border-radius: 8px !important;
                transition: all 0.2s ease !important;
            }
            
            .q-field textarea {
                resize: none !important;
            }
            
            body.dark .q-field__control {
                color: #e5e7eb !important;
            }
            
            body.dark .q-field__label {
                color: #9ca3af !important;
            }
            
            /* Primary color focus for sidebar inputs */
            .q-drawer .q-field--focused .q-field__control {
                box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.2) !important;
            }
            
            .q-drawer .q-field--focused .q-field__label {
                color: #FF4B4B !important;
            }
            
            /* Select dropdown with primary accent */
            .q-drawer .q-select .q-field__append {
                color: #FF4B4B !important;
            }
            
            /* Tabs */
            .q-tabs {
                border-bottom: 1px solid #e6e9ef !important;
            }
            
            body.dark .q-tabs {
                border-bottom-color: #262c36 !important;
            }
            
            .q-tab {
                text-transform: none !important;
                font-weight: 500 !important;
            }
            
            body.dark .q-tab {
                color: #9ca3af !important;
            }
            
            body.dark .q-tab--active {
                color: #FF4B4B !important;
            }
            
            /* Enhanced Drawer with fixed layout */
            .q-drawer {
                border-right: 1px solid #e6e9ef !important;
                width: 325px !important;
            }
            
            /* Ensure drawer slides completely off-screen when closed */
            .q-drawer--left.q-drawer--closed {
                transform: translateX(-100%) !important;
            }
            
            .q-drawer--left.q-drawer--opened {
                transform: translateX(0) !important;
            }
            
            .q-drawer__content {
                display: flex !important;
                flex-direction: column !important;
                height: 100% !important;
                overflow: hidden !important;
            }
            
            /* Fixed header and footer in sidebar - transparent to show gradient */
            .q-drawer .q-row[style*="position: sticky"][style*="top: 0"] {
                background: transparent !important;
            }
            
            .q-drawer .q-row[style*="position: sticky"][style*="bottom: 0"] {
                background: rgba(249, 250, 251, 0.95) !important;
            }
            
            body.dark .q-drawer .q-row[style*="position: sticky"][style*="bottom: 0"] {
                background: rgba(17, 24, 39, 0.95) !important;
            }
            

            
            body.dark .q-drawer {
                border-right-color: #262c36 !important;
                background: linear-gradient(180deg, #111827 0%, #0f172a 100%) !important;
            }
            

            
            /* Drawer backdrop on mobile */
            @media (max-width: 1023px) {
                .q-drawer--mobile.q-drawer--opened ~ .q-drawer__backdrop {
                    background: rgba(0, 0, 0, 0.5) !important;
                    backdrop-filter: blur(4px) !important;
                }
            }
            
            /* Dark mode for gradient card backgrounds */
            body.dark [style*="background: linear-gradient(135deg, rgba(255, 75, 75"] {
                background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%) !important;
            }
            
            body.dark [style*="background: linear-gradient(135deg, #ffffff 0%, #fef2f2"] {
                background: linear-gradient(135deg, #1f2937 0%, #1e293b 100%) !important;
            }
            
            body.dark [style*="background: linear-gradient(to top, rgba(255,255,255"] {
                background: linear-gradient(to top, rgba(17, 24, 39, 0.98), rgba(17, 24, 39, 0.95)) !important;
            }
            
            body.dark [style*="background: linear-gradient(0deg, rgba(255,75,75,0.03)"] {
                background: linear-gradient(0deg, rgba(255,75,75,0.08) 0%, transparent 100%) !important;
            }
            
            /* Dark mode for special styled cards */
            body.dark [style*="border-left: 4px solid #FF4B4B"] {
                background-color: #1a1d24 !important;
            }
            
            /* Ensure proper contrast for all text on colored backgrounds */
            body.dark [class*="bg-primary"] {
                color: #ffffff !important;
            }
            
            body.dark [class*="bg-primary"] * {
                color: #ffffff !important;
            }
            

            
            /* Ensure scrollable area maintains fixed height */
            .custom-scrollbar {
                flex: 1 1 auto !important;
                min-height: 0 !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
            }
            
            /* Prevent expansion items from affecting parent height */
            .q-drawer .q-expansion-item {
                flex-shrink: 0 !important;
            }
            
            /* Header */
            .q-header {
                border-bottom: 1px solid #e6e9ef !important;
                box-shadow: none !important;
            }
            
            body.dark .q-header {
                border-bottom-color: #262c36 !important;
                background-color: #0e1117 !important;
            }
            
            /* Enhanced Expansion panels with modern design */
            .q-expansion-item {
                margin-bottom: 12px !important;
                border-radius: 12px !important;
                overflow: visible !important;
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            .q-expansion-item__container {
                border-radius: 12px !important;
                overflow: hidden !important;
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            /* Smooth animation for expansion content */
            .q-expansion-item__content {
                overflow: hidden !important;
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            /* Add smooth slide animation using Quasar's transition classes */
            .q-expansion-item .q-item__section--main {
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            /* Enhance Quasar's built-in slide transition */
            .q-expansion-item--popup .q-expansion-item__container,
            .q-expansion-item__container > .q-card {
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            /* Smooth height animation for content wrapper */
            .q-expansion-item__content > div {
                transition: opacity 0.6s ease-in-out, transform 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            /* Center expansion item titles vertically */
            .q-drawer .q-expansion-item .q-item {
                padding: 12px 16px !important;
                min-height: 52px !important;
                background: transparent !important;
                transition: background-color 0.2s ease !important;
            }
            
            .q-drawer .q-expansion-item .q-item:hover {
                background-color: rgba(255, 75, 75, 0.05) !important;
            }
            
            body.dark .q-drawer .q-expansion-item .q-item:hover {
                background-color: rgba(255, 75, 75, 0.1) !important;
            }
            
            .q-drawer .q-expansion-item .q-item__section--avatar {
                padding-right: 12px !important;
                align-self: center !important;
                color: #FF4B4B !important;
                transition: transform 0.2s ease !important;
            }
            
            .q-drawer .q-expansion-item:hover .q-item__section--avatar {
                transform: scale(1.1) !important;
            }
            
            .q-drawer .q-expansion-item .q-item__section--main {
                align-self: center !important;
            }
            
            .q-drawer .q-expansion-item .q-item__section--side {
                align-self: center !important;
            }
            
            /* Expansion item header styling with smooth icon rotation */
            .q-expansion-item__toggle-icon {
                color: #9ca3af !important;
                transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1), 
                            color 0.3s ease !important;
            }
            
            .q-expansion-item--expanded .q-expansion-item__toggle-icon {
                transform: rotate(180deg) !important;
                color: #FF4B4B !important;
            }
            
            .q-expansion-item:hover .q-expansion-item__toggle-icon {
                color: #FF4B4B !important;
            }
            
            /* Badge styling in sidebar with primary color */
            .q-badge {
                font-weight: 600 !important;
                padding: 4px 8px !important;
                font-size: 0.75rem !important;
                box-shadow: 0 2px 4px rgba(255, 75, 75, 0.2) !important;
            }
            
            .q-drawer .q-expansion-item--expanded .q-item {
                background: transparent !important;
            }
            
            /* Labels and text - comprehensive dark mode */
            body.dark .q-item__label,
            body.dark label,
            body.dark .q-field__label,
            body.dark .q-field__native,
            body.dark .q-field__control,
            body.dark span,
            body.dark p,
            body.dark div {
                color: #e5e7eb !important;
            }
            
            /* Ensure all text inputs have proper dark mode colors */
            body.dark input,
            body.dark textarea {
                color: #e5e7eb !important;
                background-color: #1f2937 !important;
            }
            
            /* Select dropdowns */
            body.dark .q-select__dropdown-icon {
                color: #9ca3af !important;
            }
            
            body.dark .q-select .q-field__native {
                color: #e5e7eb !important;
            }
            
            /* Video player */
            body.dark video {
                border: 1px solid #262c36 !important;
            }
            
            /* Ensure all containers have proper dark backgrounds */
            body.dark .q-page,
            body.dark .q-page-container {
                background-color: #0e1117 !important;
            }
            
            /* Tab panels dark mode */
            body.dark .q-tab-panel {
                background-color: transparent !important;
                color: #e5e7eb !important;
            }
            
            /* Ensure log/code elements have dark backgrounds */
            body.dark .q-log,
            body.dark pre,
            body.dark code {
                background-color: #000000 !important;
                color: #e5e7eb !important;
                border-color: #262c36 !important;
            }
            
            /* Linear progress bars in dark mode */
            body.dark .q-linear-progress {
                background-color: #374151 !important;
            }
            
            /* Badges in dark mode */
            body.dark .q-badge {
                background-color: #374151 !important;
                color: #e5e7eb !important;
            }
            
            /* Spinners and loading indicators */
            body.dark .q-spinner {
                color: #FF4B4B !important;
            }
            
            /* Tooltips in dark mode */
            body.dark .q-tooltip {
                background-color: #1f2937 !important;
                color: #e5e7eb !important;
            }
            
            /* Ensure placeholder text is visible in dark mode */
            body.dark input::placeholder,
            body.dark textarea::placeholder {
                color: #6b7280 !important;
            }
            
            /* JSON editor in dark mode */
            body.dark .jsoneditor {
                background-color: #1f2937 !important;
                border-color: #374151 !important;
            }
            
            body.dark .jsoneditor-menu {
                background-color: #111827 !important;
                border-color: #374151 !important;
            }
            
            /* Expansion item content in dark mode */
            body.dark .q-expansion-item__content {
                background-color: transparent !important;
            }
            
            /* Ensure all icon colors are visible in dark mode */
            body.dark .q-icon {
                color: inherit !important;
            }
            
            /* Fix for any remaining white backgrounds in dark mode */
            body.dark [style*="background-color: white"],
            body.dark [style*="background: white"] {
                background-color: #1a1d24 !important;
            }
            
            /* Fix for gradient backgrounds in dark mode */
            body.dark [style*="background: linear-gradient"][style*="white"] {
                background: linear-gradient(135deg, #1a1d24 0%, #0f172a 100%) !important;
            }
            
            /* Focus states for accessibility */
            .q-btn:focus-visible {
                outline: 2px solid #FF4B4B !important;
                outline-offset: 2px !important;
            }
            
            /* Remove double border on inputs - keep only the darker inner border */
            .q-field:focus-within {
                box-shadow: none !important;
            }
            
            .q-field--outlined:focus-within .q-field__control:before {
                border-color: #FF4B4B !important;
                border-width: 2px !important;
            }
            
            .q-field--outlined:focus-within .q-field__control:after {
                display: none !important;
            }
            
            /* Also fix for select dropdowns */
            .q-select:focus-within .q-field__control:after {
                display: none !important;
            }
            
            /* Fix for textarea */
            .q-field--outlined.q-field--focused .q-field__control:after {
                display: none !important;
            }
            
            /* Skip to main content link */
            .skip-link {
                position: absolute;
                top: -40px;
                left: 0;
                background: #FF4B4B;
                color: white;
                padding: 8px;
                text-decoration: none;
                z-index: 100;
            }
            
            .skip-link:focus {
                top: 0;
            }
            
            /* Enhanced Scrollbar styling with primary color */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: transparent;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #cbd5e1;
                border-radius: 4px;
                transition: background 0.2s ease;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #94a3b8;
            }
            
            body.dark ::-webkit-scrollbar-thumb {
                background: #475569;
            }
            
            body.dark ::-webkit-scrollbar-thumb:hover {
                background: #64748b;
            }
            
            /* Custom scrollbar for sidebar with primary accent */
            .custom-scrollbar::-webkit-scrollbar {
                width: 6px;
            }
            
            .custom-scrollbar::-webkit-scrollbar-track {
                background: rgba(0, 0, 0, 0.05);
                border-radius: 3px;
                margin: 4px 0;
            }
            
            .custom-scrollbar::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #FF4B4B 0%, #ff6b6b 100%);
                border-radius: 3px;
                transition: all 0.2s ease;
            }
            
            .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(180deg, #ff3333 0%, #FF4B4B 100%);
                box-shadow: 0 0 6px rgba(255, 75, 75, 0.5);
            }
            
            body.dark .custom-scrollbar::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.05);
            }
            
            body.dark .custom-scrollbar::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #FF4B4B 0%, #cc3d3d 100%);
            }
            
            body.dark .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(180deg, #ff5555 0%, #FF4B4B 100%);
                box-shadow: 0 0 8px rgba(255, 75, 75, 0.6);
            }
            
            /* Enhanced Responsive improvements */
            @media (max-width: 1023px) {
                .q-header {
                    padding: 0 16px !important;
                }
                
                .q-drawer {
                    width: 300px !important;
                    max-width: 85vw !important;
                }
                
                body {
                    font-size: 14px !important;
                }
                
                .q-card {
                    margin: 8px !important;
                    border-radius: 8px !important;
                }
                
                /* Hide subtitle on mobile */
                .hidden.md\\:block {
                    display: none !important;
                }
            }
            
            @media (max-width: 640px) {
                .q-drawer {
                    width: 280px !important;
                }
                
                .q-header {
                    height: 56px !important;
                }
                
                /* Smaller text on very small screens */
                .text-2xl {
                    font-size: 1.25rem !important;
                }
                
                .text-xl {
                    font-size: 1.125rem !important;
                }
            }
            
            /* Tablet optimizations */
            @media (min-width: 768px) and (max-width: 1023px) {
                .q-drawer {
                    width: 325px !important;
                }
            }
            
            /* Loading states */
            .q-btn--loading {
                pointer-events: none;
                opacity: 0.7;
            }
            
            /* Empty states */
            .empty-state {
                padding: 48px 24px;
                text-align: center;
                color: #6b7280;
            }
            
            body.dark .empty-state {
                color: #9ca3af;
            }
            
            /* Log output */
            body.dark .nicegui-log {
                background-color: #000000 !important;
                border: 1px solid #262c36 !important;
            }
            
            /* Markdown content in dark mode */
            body.dark .q-markdown {
                color: #e5e7eb !important;
            }
            
            body.dark .q-markdown h1,
            body.dark .q-markdown h2,
            body.dark .q-markdown h3,
            body.dark .q-markdown h4,
            body.dark .q-markdown h5,
            body.dark .q-markdown h6 {
                color: #f3f4f6 !important;
            }
            
            body.dark .q-markdown code {
                background-color: #1f2937 !important;
                color: #e5e7eb !important;
            }
            
            body.dark .q-markdown pre {
                background-color: #1f2937 !important;
                border: 1px solid #374151 !important;
            }
            
            /* Context label dark mode */
            body.dark .bg-blue-50 {
                background-color: #1e3a5f !important;
            }
            
            body.dark .bg-green-50 {
                background-color: #1e4d2b !important;
            }
            
            body.dark .bg-orange-50 {
                background-color: #7c2d12 !important;
            }
            
            body.dark .bg-gray-50 {
                background-color: #1f2937 !important;
            }
            
            body.dark .bg-gray-100 {
                background-color: #374151 !important;
            }
            
            body.dark .bg-gray-200 {
                background-color: #4b5563 !important;
            }
            
            body.dark .border-blue-500 {
                border-color: #3b82f6 !important;
            }
            
            body.dark .border-green-500 {
                border-color: #10b981 !important;
            }
            
            body.dark .border-orange-200 {
                border-color: #ea580c !important;
            }
            
            body.dark .border-orange-800 {
                border-color: #9a3412 !important;
            }
            
            /* Text colors in dark mode */
            body.dark .text-gray-500 {
                color: #9ca3af !important;
            }
            
            body.dark .text-gray-600 {
                color: #9ca3af !important;
            }
            
            body.dark .text-gray-700 {
                color: #d1d5db !important;
            }
            
            body.dark .text-gray-900 {
                color: #f3f4f6 !important;
            }
            
            body.dark .text-orange-700 {
                color: #fdba74 !important;
            }
            
            body.dark .text-orange-300 {
                color: #fed7aa !important;
            }
            
            body.dark .text-orange-500 {
                color: #f97316 !important;
            }
            
            body.dark .text-orange-600 {
                color: #ea580c !important;
            }
            
            body.dark .text-amber-900 {
                color: #fef3c7 !important;
            }
            
            body.dark .text-amber-600 {
                color: #fbbf24 !important;
            }
            
            body.dark .text-blue-500 {
                color: #3b82f6 !important;
            }
            
            body.dark .text-green-600 {
                color: #10b981 !important;
            }
            
            body.dark .text-green-700 {
                color: #34d399 !important;
            }
            
            body.dark .text-green-300 {
                color: #6ee7b7 !important;
            }
            
            body.dark .text-red-600 {
                color: #ef4444 !important;
            }
            
            body.dark .text-red-400 {
                color: #f87171 !important;
            }
            
            /* Ensure white text stays white in dark mode */
            body.dark .text-white {
                color: #ffffff !important;
            }
            
            /* Background colors for special cards */
            body.dark [style*="background-color: #fffbeb"] {
                background-color: #422006 !important;
            }
            
            body.dark [style*="border: 1px solid #fef3c7"] {
                border-color: #92400e !important;
            }
            
            /* Green success backgrounds */
            body.dark .bg-green-900\/20 {
                background-color: rgba(6, 78, 59, 0.2) !important;
            }
            
            /* Orange warning backgrounds */
            body.dark .bg-orange-900\/20 {
                background-color: rgba(124, 45, 18, 0.2) !important;
            }
        </style>
    ''')
    
    # Add custom CSS for chat messages
    ui.add_head_html('''
    <style>
        /* Remove all message pointers/triangles */
        .q-message-text::before,
        .q-message-text::after,
        .q-message-text-content::before,
        .q-message-text-content::after {
            display: none !important;
        }
        
        /* User messages - align right with primary color */
        .q-message-sent {
            justify-content: flex-end !important;
            flex-direction: row !important;
        }
        
        .q-message-sent .q-message-container {
            flex-direction: row-reverse !important;
            align-items: flex-start !important;
        }
        
        .q-message-sent .q-message-avatar {
            margin-left: 12px !important;
            margin-right: 0 !important;
        }
        
        .q-message-sent .q-message-text {
            background-color: var(--q-primary) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
        }
        
        .q-message-sent .q-message-text p,
        .q-message-sent .q-message-text * {
            color: white !important;
        }
        
        /* AI messages - align left with gray color */
        .q-message-received {
            justify-content: flex-start !important;
            flex-direction: row !important;
        }
        
        .q-message-received .q-message-container {
            flex-direction: row !important;
            align-items: flex-start !important;
        }
        
        .q-message-received .q-message-avatar {
            margin-right: 12px !important;
            margin-left: 0 !important;
        }
        
        .q-message-received .q-message-text {
            background-color: #e5e7eb !important;
            color: #1f2937 !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
        }
        
        .q-message-received .q-message-text p,
        .q-message-received .q-message-text * {
            color: #1f2937 !important;
        }
        
        /* Dark mode adjustments */
        body.dark .q-message-received .q-message-text,
        .dark .q-message-received .q-message-text,
        html.dark .q-message-received .q-message-text {
            background-color: #374151 !important;
            color: #e5e7eb !important;
        }
        
        body.dark .q-message-received .q-message-text p,
        body.dark .q-message-received .q-message-text *,
        body.dark .q-message-received .q-message-text h1,
        body.dark .q-message-received .q-message-text h2,
        body.dark .q-message-received .q-message-text h3,
        body.dark .q-message-received .q-message-text h4,
        body.dark .q-message-received .q-message-text h5,
        body.dark .q-message-received .q-message-text h6,
        body.dark .q-message-received .q-message-text li,
        body.dark .q-message-received .q-message-text span,
        body.dark .q-message-received .q-message-text div,
        .dark .q-message-received .q-message-text p,
        .dark .q-message-received .q-message-text *,
        html.dark .q-message-received .q-message-text p,
        html.dark .q-message-received .q-message-text * {
            color: #e5e7eb !important;
        }
        
        body.dark .q-message-received .q-message-text code,
        .dark .q-message-received .q-message-text code,
        html.dark .q-message-received .q-message-text code {
            background-color: rgba(0, 0, 0, 0.3) !important;
            color: #e5e7eb !important;
        }
        
        /* Ensure sent messages stay readable */
        body.dark .q-message-sent .q-message-text,
        .dark .q-message-sent .q-message-text,
        html.dark .q-message-sent .q-message-text {
            background-color: var(--q-primary) !important;
            color: white !important;
        }
        
        body.dark .q-message-sent .q-message-text code,
        .dark .q-message-sent .q-message-text code,
        html.dark .q-message-sent .q-message-text code {
            background-color: rgba(0, 0, 0, 0.2) !important;
            color: white !important;
        }
        
        /* Avatar positioning */
        .q-message-avatar {
            min-width: 40px !important;
            flex-shrink: 0 !important;
        }
        
        /* Message name labels */
        .q-message-name {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            margin-bottom: 4px !important;
            opacity: 0.7 !important;
        }
        
        .q-message-sent .q-message-name {
            text-align: right !important;
        }
        
        .q-message-received .q-message-name {
            text-align: left !important;
        }
        
        /* Ensure messages don't have weird spacing */
        .q-message {
            margin-bottom: 16px !important;
        }
        
        .q-message-text-content {
            border-radius: 12px !important;
        }
        
        /* Fix font sizes in chat messages - make everything consistent */
        .q-message-text h1,
        .q-message-text h2,
        .q-message-text h3,
        .q-message-text h4,
        .q-message-text h5,
        .q-message-text h6 {
            font-size: 1rem !important;
            font-weight: 600 !important;
            margin: 0.5rem 0 !important;
        }
        
        .q-message-text p,
        .q-message-text li,
        .q-message-text span,
        .q-message-text div {
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
        }
        
        .q-message-text strong,
        .q-message-text b {
            font-weight: 600 !important;
        }
        
        .q-message-text code {
            font-size: 0.9rem !important;
            background-color: rgba(0, 0, 0, 0.05) !important;
            padding: 0.2rem 0.4rem !important;
            border-radius: 3px !important;
        }
        
        .q-message-text ul,
        .q-message-text ol {
            margin: 0.5rem 0 !important;
            padding-left: 1.5rem !important;
        }
        
        /* Project View enhancements */
        .prose {
            max-width: none !important;
        }
        
        .prose h1, .prose h2, .prose h3 {
            color: inherit !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        .prose p {
            margin-bottom: 1rem !important;
        }
        
        .prose code {
            background-color: rgba(0, 0, 0, 0.05) !important;
            padding: 0.2rem 0.4rem !important;
            border-radius: 3px !important;
            font-size: 0.9em !important;
        }
        
        body.dark .prose,
        body.dark .prose p,
        body.dark .prose li,
        body.dark .prose span,
        body.dark .prose div {
            color: #e5e7eb !important;
        }
        
        body.dark .prose h1,
        body.dark .prose h2,
        body.dark .prose h3,
        body.dark .prose h4,
        body.dark .prose h5,
        body.dark .prose h6 {
            color: #f3f4f6 !important;
        }
        
        body.dark .prose code {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #e5e7eb !important;
        }
        
        body.dark .prose pre {
            background-color: #1f2937 !important;
            border: 1px solid #374151 !important;
        }
        
        body.dark .prose a {
            color: #60a5fa !important;
        }
        
        body.dark .prose strong,
        body.dark .prose b {
            color: #f3f4f6 !important;
        }
        
        body.dark .prose blockquote {
            border-left-color: #4b5563 !important;
            color: #d1d5db !important;
        }
        
        body.dark .prose ul,
        body.dark .prose ol {
            color: #e5e7eb !important;
        }
        
        body.dark .prose hr {
            border-color: #374151 !important;
        }
        
        body.dark .prose table {
            border-color: #374151 !important;
        }
        
        body.dark .prose th,
        body.dark .prose td {
            border-color: #374151 !important;
            color: #e5e7eb !important;
        }
        
        body.dark .prose thead {
            background-color: #1f2937 !important;
        }
        
        /* Video player enhancements */
        video {
            outline: none !important;
        }
        
        video::-webkit-media-controls-panel {
            background-image: linear-gradient(transparent, rgba(0,0,0,0.5)) !important;
        }
        
        /* Comprehensive dark mode text color fix - AGGRESSIVE OVERRIDES */
        body.dark * {
            border-color: #374151;
        }
        
        /* Force all text elements to be light in dark mode */
        body.dark label:not(.q-btn__content *),
        body.dark span:not(.q-btn__content *):not([class*="text-white"]),
        body.dark p:not(.q-btn__content *),
        body.dark div:not(.q-btn):not([class*="bg-primary"]):not([style*="background: linear-gradient(135deg, #FF4B4B"]) {
            color: #e5e7eb !important;
        }
        
        /* Override any inline text colors in dark mode */
        body.dark [style*="color: rgb(31, 41, 55)"]:not(.q-btn__content *),
        body.dark [style*="color: rgb(55, 65, 81)"]:not(.q-btn__content *),
        body.dark [style*="color: rgb(75, 85, 99)"]:not(.q-btn__content *),
        body.dark [style*="color: rgb(107, 114, 128)"]:not(.q-btn__content *),
        body.dark [style*="color: #1f2937"]:not(.q-btn__content *),
        body.dark [style*="color: #374151"]:not(.q-btn__content *),
        body.dark [style*="color: #4b5563"]:not(.q-btn__content *),
        body.dark [style*="color: #6b7280"]:not(.q-btn__content *) {
            color: #e5e7eb !important;
        }
        
        /* Force all card backgrounds to be dark */
        body.dark .q-card:not([style*="background: linear-gradient(135deg, #FF4B4B"]):not([style*="background: linear-gradient(135deg, rgba(255, 75, 75"]) {
            background-color: #1a1d24 !important;
        }
        
        /* Override white backgrounds */
        body.dark [style*="background-color: white"]:not(.q-btn),
        body.dark [style*="background-color: #ffffff"]:not(.q-btn),
        body.dark [style*="background-color: rgb(255, 255, 255)"]:not(.q-btn),
        body.dark [style*="background: white"]:not(.q-btn) {
            background-color: #1a1d24 !important;
        }
        
        /* Force page backgrounds */
        body.dark .q-page,
        body.dark .q-page-container,
        body.dark .q-layout__section {
            background-color: #0e1117 !important;
        }
        
        body.dark .q-field__control::before {
            border-color: #4b5563 !important;
        }
        
        body.dark .q-field__control::after {
            border-color: #FF4B4B !important;
        }
        
        /* Ensure all avatars and icons maintain visibility */
        body.dark .q-avatar {
            background-color: #374151 !important;
        }
        
        /* Override specific inline styled elements */
        body.dark [style*="padding: 60px; text-align: center"] {
            background-color: #1a1d24 !important;
        }
        
        body.dark [style*="padding: 24px"] .q-card,
        body.dark [style*="padding: 32px"] {
            background-color: #1a1d24 !important;
        }
        
        /* Fix for gradient header cards */
        body.dark [style*="background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%)"] * {
            color: white !important;
        }
        
        /* Fix for chat container backgrounds */
        body.dark [style*="background: linear-gradient(to top, rgba(255,255,255"] {
            background: linear-gradient(to top, rgba(17, 24, 39, 0.98), rgba(17, 24, 39, 0.95)) !important;
        }
        
        /* Fix for sidebar footer gradient */
        body.dark [style*="background: linear-gradient(0deg, rgba(255,75,75,0.03)"] {
            background: linear-gradient(0deg, rgba(255,75,75,0.08) 0%, transparent 100%) !important;
        }
        
        /* Fix for welcome card gradient */
        body.dark [style*="background: linear-gradient(135deg, #ffffff 0%, #fef2f2"] {
            background: linear-gradient(135deg, #1f2937 0%, #1e293b 100%) !important;
        }
        
        /* Fix for typing indicator gradient */
        body.dark [style*="background: linear-gradient(135deg, rgba(255, 75, 75, 0.05) 0%, rgba(99, 102, 241, 0.05)"] {
            background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%) !important;
        }
        
        /* Fix for tips card */
        body.dark [style*="background-color: #fffbeb"] {
            background-color: #422006 !important;
        }
        
        body.dark [style*="border: 1px solid #fef3c7"] {
            border-color: #92400e !important;
        }
        
        /* Fix for all text in special background cards */
        body.dark [style*="background-color: #fffbeb"] *,
        body.dark [class*="bg-orange-50"] *,
        body.dark [class*="bg-green-50"] *,
        body.dark [class*="bg-blue-50"] * {
            color: #e5e7eb !important;
        }
        
        /* Fix for any inline styles that might override dark mode */
        body.dark [style*="color: #1f2937"],
        body.dark [style*="color: #374151"],
        body.dark [style*="color: #4b5563"],
        body.dark [style*="color: #6b7280"] {
            color: #e5e7eb !important;
        }
        
        /* Ensure separators are visible in dark mode */
        body.dark .q-separator {
            background-color: #374151 !important;
        }
        
        /* Fix for notification/toast messages */
        body.dark .q-notification {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
            border: 1px solid #374151 !important;
        }
        
        body.dark .q-notification__message {
            color: #e5e7eb !important;
        }
        
        /* Ensure menu items are visible */
        body.dark .q-menu {
            background-color: #1f2937 !important;
        }
        
        body.dark .q-item {
            color: #e5e7eb !important;
        }
        
        body.dark .q-item:hover {
            background-color: #374151 !important;
        }
        
        /* Fix for dialog backgrounds */
        body.dark .q-dialog__backdrop {
            background-color: rgba(0, 0, 0, 0.7) !important;
        }
        
        /* Ensure all labels in forms are visible */
        body.dark .q-field__label,
        body.dark .q-field__native,
        body.dark .q-field__prefix,
        body.dark .q-field__suffix {
            color: #e5e7eb !important;
        }
        
        /* Fix for empty state text */
        body.dark .empty-state {
            color: #9ca3af !important;
        }
        
        /* Ensure all headings are visible */
        body.dark h1,
        body.dark h2,
        body.dark h3,
        body.dark h4,
        body.dark h5,
        body.dark h6 {
            color: #f3f4f6 !important;
        }
        
        /* NUCLEAR OPTION - Force all non-button text to be light */
        body.dark *:not(.q-btn):not(.q-btn *):not([class*="text-white"]):not([class*="text-primary"]):not([class*="text-red"]):not([class*="text-green"]):not([class*="text-blue"]):not([class*="text-orange"]):not([class*="text-amber"]) {
            color: #e5e7eb !important;
        }
        
        /* Force all non-gradient, non-button backgrounds to be dark */
        body.dark .q-card:not([style*="gradient"]),
        body.dark [class*="card"]:not([style*="gradient"]):not(.q-btn) {
            background-color: #1a1d24 !important;
            border-color: #262c36 !important;
        }
        
        /* Ensure input fields are always dark */
        body.dark .q-field__control,
        body.dark .q-field__native,
        body.dark input,
        body.dark textarea,
        body.dark select {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
        }
        
        /* Fix for any remaining light backgrounds */
        body.dark [style*="background-color: #f"]:not(.q-btn):not([style*="background-color: #FF4B4B"]),
        body.dark [style*="background-color: #e"]:not(.q-btn),
        body.dark [style*="background-color: #d"]:not(.q-btn),
        body.dark [style*="background-color: #c"]:not(.q-btn),
        body.dark [style*="background-color: #b"]:not(.q-btn),
        body.dark [style*="background-color: #a"]:not(.q-btn),
        body.dark [style*="background-color: #9"]:not(.q-btn),
        body.dark [style*="background-color: #8"]:not(.q-btn) {
            background-color: #1a1d24 !important;
        }
        
        /* Ensure all containers have dark backgrounds */
        body.dark .q-panel,
        body.dark .q-tab-panel,
        body.dark .q-expansion-item__container {
            background-color: transparent !important;
        }
        
        /* Fix for grid and column layouts */
        body.dark .q-grid,
        body.dark [class*="grid"],
        body.dark .q-column,
        body.dark [class*="column"] {
            background-color: transparent !important;
        }
    </style>
    ''')
    
    dark_mode = ui.dark_mode(value=False)
    
    # Add comprehensive JavaScript to enforce dark mode
    ui.add_body_html('''
    <script>
    // Comprehensive dark mode enforcement with proper detection
    function enforceDarkMode() {
        // Check multiple possible dark mode indicators
        const isDark = document.body.classList.contains('dark') || 
                      document.body.classList.contains('body--dark') ||
                      document.documentElement.classList.contains('dark');
        
        console.log('Dark mode active:', isDark);
        
        if (isDark) {
            // Force page background
            document.body.style.backgroundColor = '#0e1117';
            
            // Force all text to be light colored
            document.querySelectorAll('label, span, p, div, h1, h2, h3, h4, h5, h6').forEach(el => {
                // Skip buttons and elements that should stay their color
                if (el.closest('.q-btn--unelevated') || 
                    el.classList.contains('text-white') ||
                    el.closest('[style*="background: linear-gradient(135deg, #FF4B4B"]')) {
                    return;
                }
                
                const computedStyle = window.getComputedStyle(el);
                const color = computedStyle.color;
                
                // Check if text is dark
                if (color.startsWith('rgb')) {
                    const match = color.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    if (match) {
                        const r = parseInt(match[1]);
                        const g = parseInt(match[2]);
                        const b = parseInt(match[3]);
                        const brightness = (r + g + b) / 3;
                        
                        // If text is dark (brightness < 150), make it light
                        if (brightness < 150) {
                            el.style.setProperty('color', '#e5e7eb', 'important');
                        }
                    }
                }
            });
            
            // Force card backgrounds
            document.querySelectorAll('.q-card').forEach(card => {
                const style = window.getComputedStyle(card);
                const bg = style.backgroundColor;
                
                // Skip gradient backgrounds
                if (card.style.background && card.style.background.includes('gradient')) {
                    if (card.style.background.includes('#FF4B4B')) {
                        return; // Keep primary gradient
                    }
                }
                
                // Check if background is light
                if (bg.startsWith('rgb')) {
                    const match = bg.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    if (match) {
                        const r = parseInt(match[1]);
                        const g = parseInt(match[2]);
                        const b = parseInt(match[3]);
                        const brightness = (r + g + b) / 3;
                        
                        if (brightness > 100) {
                            card.style.setProperty('background-color', '#1a1d24', 'important');
                            card.style.setProperty('border-color', '#262c36', 'important');
                        }
                    }
                }
            });
            
            // Force input fields
            document.querySelectorAll('input, textarea, select').forEach(input => {
                input.style.setProperty('color', '#e5e7eb', 'important');
                input.style.setProperty('background-color', '#1f2937', 'important');
            });
            
            // Force page containers
            document.querySelectorAll('.q-page, .q-page-container, .q-layout__section').forEach(el => {
                el.style.setProperty('background-color', '#0e1117', 'important');
            });
            
            // Force field controls
            document.querySelectorAll('.q-field__control').forEach(el => {
                el.style.setProperty('background-color', '#1f2937', 'important');
            });
        } else {
            // Light mode - remove forced styles
            document.querySelectorAll('[style*="color: rgb(229, 231, 235)"]').forEach(el => {
                el.style.removeProperty('color');
            });
        }
    }
    
    // Run enforcement multiple times to catch all elements
    setTimeout(enforceDarkMode, 50);
    setTimeout(enforceDarkMode, 200);
    setTimeout(enforceDarkMode, 500);
    setTimeout(enforceDarkMode, 1000);
    setTimeout(enforceDarkMode, 2000);
    
    // Watch for dark mode class changes on body and html
    const bodyObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class') {
                console.log('Class changed on', mutation.target.tagName);
                setTimeout(enforceDarkMode, 10);
                setTimeout(enforceDarkMode, 100);
            }
        });
    });
    
    bodyObserver.observe(document.body, {
        attributes: true,
        attributeFilter: ['class']
    });
    
    bodyObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['class']
    });
    
    // Watch for new elements being added
    const contentObserver = new MutationObserver(() => {
        setTimeout(enforceDarkMode, 50);
    });
    
    contentObserver.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // Also trigger on window load
    window.addEventListener('load', () => {
        setTimeout(enforceDarkMode, 100);
        setTimeout(enforceDarkMode, 500);
    });
    </script>
    ''')

    # --- Modern Header with Enhanced Styling ---
    with ui.header().classes(
        "bg-white dark:bg-gray-900 text-gray-900 dark:text-white shadow-sm"
    ).style("height: 64px; padding: 0 24px; display: flex; align-items: center; backdrop-filter: blur(10px);"):
        with ui.row().classes("w-full items-center justify-between").style("height: 100%;"):
            with ui.row(align_items="center").classes("gap-3"):
                # Enhanced menu button with hover effect
                menu_btn = ui.button(
                    icon="menu",
                    on_click=lambda: left_drawer.toggle()
                ).props("flat round").classes(
                    "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                ).tooltip("Toggle sidebar")
                
                # Animated brand section
                with ui.row(align_items="center").classes("gap-2"):
                    ui.icon("movie", size="md").classes("text-primary")
                    ui.label("AlgoVision").classes("text-xl font-bold tracking-tight text-gray-900 dark:text-white")
            
            with ui.row(align_items="center").classes("gap-4"):
                ui.label("Educational Video Generator").classes(
                    "text-sm text-gray-500 dark:text-gray-400 hidden md:block font-medium"
                )
                
                # Enhanced dark mode toggle with icon transition
                dark_mode_btn = ui.button(
                    icon="light_mode" if dark_mode.value else "dark_mode",
                    on_click=lambda: (dark_mode.toggle(), dark_mode_btn.props(f"icon={'dark_mode' if dark_mode.value else 'light_mode'}"))
                ).props("flat round").classes(
                    "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                ).tooltip("Toggle dark mode")

    # --- Modern Sidebar (Left Drawer) with Enhanced UX ---
    left_drawer = ui.left_drawer(value=True, fixed=True, bordered=True).classes(
        "bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800"
    )
    
    with left_drawer:
        # Sidebar Header with branding and primary accent
        with ui.row().classes("w-full items-center justify-center px-4 py-3 border-b-2 border-primary/30 dark:border-primary/40").style("flex-shrink: 0; backdrop-filter: blur(10px);"):
            with ui.column().classes("gap-1 items-center text-center"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("settings", size="md").classes("text-primary")
                    ui.label("Settings").classes("text-2xl font-bold text-gray-900 dark:text-white")
                ui.label("Configure your workspace").classes("text-xs text-primary/70 dark:text-primary/60 font-medium")
        
        # Scrollable content area with custom scrollbar - fixed height to enable proper scrolling
        with ui.column().classes("w-full gap-3 px-4 pt-4 pb-2 overflow-y-auto overflow-x-hidden custom-scrollbar").style("flex: 1 1 auto; min-height: 0; max-height: 100%;"):
            
            # Model Configuration Section with primary accent
            with ui.expansion(
                "Model Configuration", 
                icon="psychology",
                value=True
            ).classes(
                "w-full bg-white dark:bg-gray-800 rounded-lg shadow-sm "
                "hover:shadow-md border border-gray-200 dark:border-gray-700"
            ).props("dense duration=600"):
                ui.separator().classes("bg-gradient-to-r from-primary/30 via-primary/10 to-transparent").style("margin: 0; height: 1px;")
                with ui.column().classes("w-full gap-3 px-3 pt-2 pb-3"):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.icon("smart_toy", size="sm").classes("text-primary")
                        ui.label("AI Model").classes("text-sm font-semibold text-primary dark:text-primary")
                    
                    def prettify_model_name(model_id: str) -> str:
                        name_part = model_id.split("/")[-1].replace("-", " ").replace("_", " ")
                        return " ".join(word.capitalize() for word in name_part.split())

                    model_display_map = {
                        model_id: prettify_model_name(model_id) for model_id in allowed_models
                    }
                    default_model = app_state.get("planner_model_name") or next(
                        iter(model_display_map)
                    )
                    
                    model_select = ui.select(
                        model_display_map,
                        label="Select LLM Model",
                        value=default_model,
                        on_change=lambda e: app_state.update({"planner_model_name": e.value}),
                    ).props("outlined dense").classes("w-full")
                    
                    # Model info badge
                    with ui.row().classes("items-center gap-2 mt-1"):
                        ui.icon("info", size="xs").classes("text-blue-500")
                        ui.label("Powers system logic & output").classes("text-xs text-gray-500 dark:text-gray-400")

            # TTS Configuration Section with primary accent on hover
            with ui.expansion(
                "Voice Configuration", 
                icon="record_voice_over",
                value=True
            ).classes(
                "w-full bg-white dark:bg-gray-800 rounded-lg shadow-sm "
                "hover:shadow-md border border-gray-200 dark:border-gray-700"
            ).props("dense duration=600"):
                ui.separator().classes("bg-gradient-to-r from-primary/30 via-primary/10 to-transparent").style("margin: 0; height: 1px;")
                with ui.column().classes("w-full gap-3 px-3 pt-2 pb-3"):
                    with ui.row().classes("items-center gap-2 mb-2"):
                        ui.icon("mic", size="sm").classes("text-primary")
                        ui.label("Voice Selection").classes("text-sm font-semibold text-primary dark:text-primary")
                    
                    voices = load_voices("voices.json")
                    if voices:
                        voice_options = {v["id"]: v["name"] for v in voices}
                        first_voice_id = next(iter(voice_options))
                        ui.select(
                            voice_options, 
                            label="Choose Voice", 
                            value=first_voice_id
                        ).props("outlined dense").classes("w-full")
                        
                        with ui.row().classes("items-center gap-1 mt-1"):
                            ui.icon("info", size="xs").classes("text-blue-500")
                            ui.label(f"{len(voices)} voices available").classes("text-xs text-gray-500 dark:text-gray-400")
                    else:
                        with ui.card().classes("w-full bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800"):
                            with ui.row().classes("items-center gap-2 p-2"):
                                ui.icon("warning", size="sm").classes("text-orange-500")
                                ui.label("No voices found").classes("text-sm text-orange-700 dark:text-orange-300")

            # Pipeline Configuration Section with primary accent on hover
            with ui.expansion(
                "Pipeline Settings", 
                icon="tune",
                value=True
            ).props("dense duration=600").classes(
                "w-full bg-white dark:bg-gray-800 rounded-lg shadow-sm "
                "hover:shadow-md border border-gray-200 dark:border-gray-700"
            ).props("dense"):
                ui.separator().classes("bg-gradient-to-r from-primary/30 via-primary/10 to-transparent").style("margin: 0; height: 1px;")
                # Max retries is now fixed at 5 (no UI control needed)
        
        # Sidebar Footer with quick actions and primary accent - fixed at bottom
        with ui.row().classes("w-full items-center justify-between px-4 py-2 border-t-2 border-primary/30 dark:border-primary/40").style("flex-shrink: 0; position: sticky; bottom: 0; z-index: 10; backdrop-filter: blur(10px); background: linear-gradient(0deg, rgba(255,75,75,0.03) 0%, transparent 100%);"):
            ui.label("v1.0.0").classes("text-xs text-gray-400")
            with ui.row().classes("gap-1"):
                ui.button(icon="help_outline", on_click=lambda: ui.notify("Documentation coming soon!", type="info")).props("flat round dense").classes("text-primary hover:bg-primary/10").tooltip("Help")
                ui.button(icon="bug_report", on_click=lambda: ui.notify("Report issues on GitHub", type="info")).props("flat round dense").classes("text-primary hover:bg-primary/10").tooltip("Report Bug")

    # --- ALL HANDLER FUNCTIONS DEFINED FIRST (will be defined after UI elements) ---

    def delete_project(topic_name):
        """Delete a project and all its files."""
        import shutil
        
        async def confirm_delete():
            dialog.close()
            try:
                project_path = get_project_path("output", topic_name)
                # Get the parent directory (the topic folder)
                topic_folder = os.path.dirname(project_path)
                
                # Delete the entire topic folder
                if os.path.exists(topic_folder):
                    shutil.rmtree(topic_folder)
                    ui.notify(f"Project '{topic_name}' deleted successfully", color="positive", icon="delete")
                    
                    # Clear chat history for this topic
                    if topic_name in app_state["chat_histories"]:
                        del app_state["chat_histories"][topic_name]
                    
                    # Clear selected topic if it was deleted
                    if app_state.get("selected_topic") == topic_name:
                        app_state["selected_topic"] = None
                        app_state["current_topic_inspector"] = None
                    
                    # Refresh dashboard and inspector
                    await update_dashboard()
                    await update_inspector(None)
                else:
                    ui.notify(f"Project folder not found: {topic_folder}", color="warning")
            except Exception as e:
                ui.notify(f"Error deleting project: {e}", color="negative", multi_line=True)
        
        # Create confirmation dialog
        with ui.dialog() as dialog, ui.card().classes("p-6"):
            with ui.column().classes("gap-4"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("warning", size="lg").classes("text-orange-500")
                    ui.label("Delete Project?").classes("text-xl font-bold")
                
                ui.label(f"Are you sure you want to delete '{topic_name}'?").classes("text-gray-700 dark:text-gray-300")
                ui.label("This will permanently delete all files, videos, and data for this project.").classes("text-sm text-gray-600 dark:text-gray-400")
                ui.label("This action cannot be undone.").classes("text-sm font-semibold text-red-600 dark:text-red-400")
                
                with ui.row().classes("w-full justify-end gap-2 mt-2"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Delete", on_click=confirm_delete).props("unelevated").classes("bg-red-600 text-white")
        
        dialog.open()

    async def update_dashboard():
        dashboard_content.clear()
        if not video_generator:
            with dashboard_content:
                ui.label("Generator not initialized.").classes("text-negative")
            return
        topic_folders = get_topic_folders("output")
        if not topic_folders:
            with dashboard_content:
                # Empty state
                with ui.card().classes("w-full").style("padding: 60px; text-align: center;"):
                    ui.icon("folder_open", size="xl").classes("text-gray-400")
                    ui.label("No Projects Yet").classes("text-2xl font-semibold text-gray-700 dark:text-gray-300 mt-4")
                    ui.label("Create your first educational video to get started").classes("text-gray-500 dark:text-gray-400 mt-2")
                    ui.button("Create Project", icon="add", on_click=lambda: main_tabs.set_value("✨ Generate")).props("unelevated no-caps").classes("mt-4")
            return
        with dashboard_content:
            all_statuses = [
                video_generator.check_theorem_status({"theorem": th})
                for th in topic_folders
            ]
            
            # Stats Cards
            with ui.grid(columns=3).classes("w-full gap-4 mb-6"):
                # Total Projects
                with ui.card().classes("w-full").style("padding: 24px;"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.column().classes("gap-1"):
                            ui.label("Total Projects").classes("text-sm font-medium text-gray-600 dark:text-gray-400")
                            ui.label(str(len(all_statuses))).classes("text-3xl font-bold text-gray-900 dark:text-white")
                        ui.icon("folder", size="lg").classes("text-primary opacity-80")
                
                # Scenes Progress
                with ui.card().classes("w-full").style("padding: 24px;"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.column().classes("gap-1"):
                            ui.label("Scenes Rendered").classes("text-sm font-medium text-gray-600 dark:text-gray-400")
                            total_scenes = sum(s["total_scenes"] for s in all_statuses)
                            total_renders = sum(s["rendered_scenes"] for s in all_statuses)
                            ui.label(f"{total_renders}/{total_scenes}").classes("text-3xl font-bold text-gray-900 dark:text-white")
                        ui.icon("movie_filter", size="lg").classes("text-blue-500 opacity-80")
                
                # Completed Videos
                with ui.card().classes("w-full").style("padding: 24px;"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.column().classes("gap-1"):
                            ui.label("Completed").classes("text-sm font-medium text-gray-600 dark:text-gray-400")
                            total_combined = sum(1 for s in all_statuses if s["has_combined_video"])
                            ui.label(f"{total_combined}/{len(all_statuses)}").classes("text-3xl font-bold text-gray-900 dark:text-white")
                        ui.icon("check_circle", size="lg").classes("text-green-500 opacity-80")
            
            # Projects List Header with View Toggle
            if "dashboard_view" not in app_state:
                app_state["dashboard_view"] = "list"
            
            view_mode = app_state["dashboard_view"]
            
            with ui.row().classes("w-full items-center justify-between mb-3"):
                ui.label("Your Projects").classes("text-lg font-semibold text-gray-900 dark:text-white")
                
                async def toggle_view():
                    app_state["dashboard_view"] = "grid" if view_mode == "list" else "list"
                    await update_dashboard()
                
                ui.button(
                    icon="grid_view" if view_mode == "list" else "view_list",
                    on_click=toggle_view
                ).props("outline dense").tooltip("Toggle Grid/List View")
            
            # Render projects based on view mode
            if view_mode == "grid":
                # Grid View
                with ui.grid(columns=2).classes("w-full gap-4"):
                    for status in sorted(all_statuses, key=lambda x: x["topic"]):
                        with ui.card().classes("w-full hover:shadow-lg transition-shadow").style("padding: 20px;"):
                            with ui.column().classes("w-full gap-3"):
                                # Header
                                with ui.row().classes("w-full items-start justify-between"):
                                    ui.icon("video_library", size="lg").classes("text-primary")
                                    with ui.row().classes("items-center gap-1"):
                                        ui.badge(
                                            "✓" if status["has_combined_video"] else "...",
                                            color="positive" if status["has_combined_video"] else "orange"
                                        ).props("rounded")
                                        ui.button(
                                            icon="delete",
                                            on_click=lambda s=status["topic"]: delete_project(s)
                                        ).props("flat round dense").classes("text-red-500").tooltip("Delete project")
                                
                                # Title
                                ui.label(status["topic"]).classes("text-lg font-semibold text-gray-900 dark:text-white")
                                
                                # Stats
                                progress_pct = int((status["rendered_scenes"] / status["total_scenes"]) * 100) if status["total_scenes"] > 0 else 0
                                with ui.column().classes("w-full gap-1"):
                                    ui.label(f"{status['rendered_scenes']}/{status['total_scenes']} scenes").classes("text-sm text-gray-600 dark:text-gray-400")
                                    ui.linear_progress(progress_pct / 100, show_value=False).props('rounded color="primary"').style("height: 6px;")
                                
                                # Action
                                ui.button(
                                    "View Details",
                                    icon="arrow_forward",
                                    on_click=lambda s=status["topic"]: asyncio.create_task(inspect_project(s))
                                ).props("flat no-caps dense").classes("w-full text-primary")
            else:
                # List View
                for status in sorted(all_statuses, key=lambda x: x["topic"]):
                    with ui.card().classes("w-full hover:shadow-lg transition-shadow").style("padding: 20px;"):
                        with ui.row().classes("w-full items-center justify-between mb-3"):
                            with ui.row().classes("items-center gap-3"):
                                ui.icon("video_library", size="md").classes("text-primary")
                                ui.label(status["topic"]).classes("text-lg font-semibold text-gray-900 dark:text-white")
                            with ui.row().classes("items-center gap-2"):
                                ui.badge(
                                    "Completed" if status["has_combined_video"] else "In Progress",
                                    color="positive" if status["has_combined_video"] else "orange"
                                ).props("rounded")
                                ui.button(
                                    icon="delete",
                                    on_click=lambda s=status["topic"]: delete_project(s)
                                ).props("flat round dense").classes("text-red-500").tooltip("Delete project")
                        
                        # Progress bar
                        progress_pct = int((status["rendered_scenes"] / status["total_scenes"]) * 100) if status["total_scenes"] > 0 else 0
                        with ui.column().classes("w-full gap-1"):
                            with ui.row().classes("w-full items-center justify-between"):
                                ui.label(f"{status['rendered_scenes']} of {status['total_scenes']} scenes").classes("text-sm text-gray-600 dark:text-gray-400")
                                ui.label(f"{progress_pct}%").classes("text-sm font-medium text-gray-700 dark:text-gray-300")
                            ui.linear_progress(progress_pct / 100, show_value=False).props('rounded color="primary"').style("height: 8px;")
                        
                        # Action button
                        ui.button(
                            "View Details",
                            icon="arrow_forward",
                            on_click=lambda s=status: asyncio.create_task(inspect_project(s["topic"]))
                        ).props("flat no-caps dense").classes("mt-2 text-primary")

    async def update_inspector(topic):
        inspector_content.clear()
        if not topic:
            with inspector_content:
                # Empty state with better design
                with ui.column().classes("w-full items-center justify-center py-20"):
                    ui.icon("video_library", size="4rem").classes("text-gray-300 dark:text-gray-600 mb-4")
                    ui.label("No Project Selected").classes("text-2xl font-semibold text-gray-400 dark:text-gray-500")
                    ui.label("Select a project from the dropdown above to view details").classes("text-gray-400 dark:text-gray-500 mt-2")
            return
        app_state["current_topic_inspector"] = topic
        with inspector_content:
            project_path = get_project_path("output", topic)
            inner_folder_name = os.path.basename(project_path)

            scene_dirs = sorted(
                [
                    d
                    for d in os.listdir(project_path)
                    if d.startswith("scene")
                    and os.path.isdir(os.path.join(project_path, d))
                ],
                key=lambda x: int(x.replace("scene", "")),
            )
            
            # Modern Project Header Card
            with ui.card().classes("w-full mb-6 shadow-lg border-l-4 border-primary"):
                with ui.row().classes("w-full items-center justify-between p-4"):
                    with ui.column().classes("gap-1"):
                        ui.label(topic).classes("text-2xl font-bold text-gray-900 dark:text-white")
                        with ui.row().classes("gap-4 items-center mt-2"):
                            with ui.row().classes("items-center gap-1"):
                                ui.icon("movie", size="sm").classes("text-gray-500")
                                ui.label(f"{len(scene_dirs)} Scenes").classes("text-sm text-gray-600 dark:text-gray-400")
                            
                            # Check if video exists
                            video_path = os.path.join(project_path, f"{inner_folder_name}_combined.mp4")
                            if os.path.exists(video_path):
                                video_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                                with ui.row().classes("items-center gap-1"):
                                    ui.icon("check_circle", size="sm").classes("text-green-500")
                                    ui.label(f"Video Ready ({video_size:.1f} MB)").classes("text-sm text-gray-600 dark:text-gray-400")
                            else:
                                with ui.row().classes("items-center gap-1"):
                                    ui.icon("pending", size="sm").classes("text-orange-500")
                                    ui.label("Video Pending").classes("text-sm text-gray-600 dark:text-gray-400")
                    
                    # Quick actions
                    with ui.row().classes("gap-2"):
                        ui.button(icon="folder_open", on_click=lambda: os.startfile(project_path)).props("flat round").tooltip("Open in Explorer")
                        ui.button(icon="refresh", on_click=lambda: asyncio.create_task(update_inspector(topic))).props("flat round").tooltip("Refresh")

            with ui.tabs().classes("w-full bg-transparent").props("dense align=left") as inspector_tabs:
                t_video = ui.tab("🎬 Player").props("no-caps")
                t_plan = ui.tab("📖 Master Plan").props("no-caps")
                scene_tabs_list = [
                    ui.tab(f"🎞️ Scene {i+1}").props("no-caps") for i in range(len(scene_dirs))
                ]

            with ui.tab_panels(inspector_tabs, value=t_video).classes(
                "w-full bg-transparent mt-6"
            ):
                with ui.tab_panel(t_video):
                    video_local_path = os.path.join(
                        project_path, f"{inner_folder_name}_combined.mp4"
                    )
                    tcm_path = os.path.join(
                        project_path, f"{inner_folder_name}_combined_tcm.json"
                    )
                    srt_local_path = os.path.join(
                        project_path, f"{inner_folder_name}_combined.srt"
                    )
                    vtt_local_path = os.path.join(
                        project_path, f"{inner_folder_name}_combined.vtt"
                    )
                    
                    # Convert SRT to VTT if needed
                    if os.path.exists(srt_local_path) and not os.path.exists(vtt_local_path):
                        with open(srt_local_path, 'r', encoding='utf-8') as f:
                            srt_content = f.read()
                        vtt_content = srt_to_vtt(srt_content)
                        with open(vtt_local_path, 'w', encoding='utf-8') as f:
                            f.write(vtt_content)
                    
                    if os.path.exists(video_local_path) and os.path.exists(tcm_path):
                        video_url = (
                            f"/{os.path.relpath(video_local_path, '.')}".replace(
                                "\\", "/"
                            )
                        )
                        
                        # Use VTT file (better browser support)
                        subtitle_url = None
                        if os.path.exists(vtt_local_path):
                            subtitle_url = f"/{os.path.relpath(vtt_local_path, '.')}".replace("\\", "/")
                        with open(tcm_path, "r", encoding="utf-8") as f:
                            tcm_data = json.load(f)
                        # Helper function to render Streamlit-style chat messages
                        def render_chat_message(container, content, is_user=False):
                            """Render a Streamlit-style chat message"""
                            with container:
                                role = "user" if is_user else "assistant"
                                avatar_text = "👤" if is_user else "🤖"
                                
                                with ui.element('div').classes(f"stchat-message {role}"):
                                    # Avatar
                                    with ui.element('div').classes("stchat-avatar"):
                                        ui.label(avatar_text).classes("text-xl")
                                    
                                    # Content
                                    with ui.element('div').classes("stchat-content"):
                                        ui.markdown(content)
                        
                        with ui.row().classes(
                            "w-full no-wrap grid grid-cols-1 lg:grid-cols-5 gap-6"
                        ):
                            with ui.column().classes("lg:col-span-3 gap-4").style("height: calc(100vh - 200px); min-height: 500px; max-height: 700px; display: flex; flex-direction: column;"):
                                # Video player card
                                with ui.card().classes("w-full shadow-2xl overflow-hidden p-0 border-0").style("flex-shrink: 0;"):
                                    # Use HTML video element with SRT subtitles
                                    video_player = None  # Initialize for later reference
                                    if subtitle_url and os.path.exists(vtt_local_path):
                                        # Create a dummy element to attach events to
                                        video_player = ui.element('div').classes('hidden')
                                        
                                        # Simple video ID
                                        video_id = "video_player"
                                        
                                        ui.html(
                                            f'''
                                            <video id="{video_id}" controls 
                                                   style="width: 100%; display: block; background: #000;">
                                                <source src="{video_url}" type="video/mp4">
                                                <track kind="subtitles" src="{subtitle_url}" srclang="en" label="English" default>
                                            </video>
                                            ''',
                                            sanitize=False
                                        )
                                        
                                        # Enable subtitles automatically
                                        def enable_subs():
                                            ui.run_javascript(f'''
                                                const video = document.getElementById('{video_id}');
                                                if (video && video.textTracks.length > 0) {{
                                                    video.textTracks[0].mode = 'showing';
                                                }}
                                            ''')
                                        
                                        # Try multiple times to ensure subtitles are enabled
                                        ui.timer(0.5, enable_subs, once=True)
                                        ui.timer(1.5, enable_subs, once=True)
                                    else:
                                        video_player = ui.video(video_url).classes(
                                            "w-full"
                                        ).style("display: block;")
                                        video_player.props('id="video_player"')
                                
                                # Quiz Section - matching AI Tutor styling
                                quiz_state = {
                                    "questions": [],
                                    "current_index": 0,
                                    "show_answer": False,
                                    "quiz_started": False
                                }
                                
                                with ui.card().classes("w-full shadow-2xl border-0").style("flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden;"):
                                    # Quiz header - matching AI Tutor header
                                    with ui.row().classes("w-full items-center justify-between px-5 py-3 border-b dark:border-gray-700").style("flex-shrink: 0; background: linear-gradient(135deg, rgba(255, 75, 75, 0.05) 0%, rgba(255, 107, 107, 0.02) 100%);"):
                                        with ui.row().classes("items-center gap-3"):
                                            with ui.avatar(size="md").classes("shadow-md").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);"):
                                                ui.icon("quiz", size="sm").classes("text-white")
                                            with ui.column().classes("gap-0"):
                                                ui.label("Interactive Quiz").classes("text-lg font-bold text-gray-900 dark:text-white")
                                                ui.label("Test your understanding").classes("text-xs text-gray-500 dark:text-gray-400")
                                        
                                    # Quiz content container - matching AI Tutor scrollable area
                                    with ui.column().style("flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; width: 100% !important; box-sizing: border-box;").classes("p-4 gap-3"):
                                        
                                        # Store quiz settings at this scope
                                        quiz_settings = {"num_questions": 5, "question_type": "Multiple Choice", "difficulty": "Medium"}
                                        
                                        # Create a container for dynamic content (this will be cleared)
                                        quiz_content_container = ui.column().classes("w-full gap-3")
                                        
                                        # Define helper functions at the proper scope
                                        def show_question():
                                            """Display the current question"""
                                            try:
                                                quiz_content_container.clear()
                                                
                                                if quiz_state["current_index"] >= len(quiz_state["questions"]):
                                                    # Quiz completed - celebration card
                                                    with quiz_content_container:
                                                        with ui.card().classes("w-full border-0 shadow-2xl").style("background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-left: 4px solid #10b981; padding: 40px;"):
                                                            with ui.column().classes("items-center gap-4 w-full"):
                                                                with ui.avatar(size="xl").classes("shadow-lg").style("background: linear-gradient(135deg, #10b981 0%, #059669 100%);"):
                                                                    ui.icon("emoji_events", size="lg").classes("text-white")
                                                                ui.label("Quiz Completed!").classes("text-2xl font-bold text-green-900")
                                                                ui.label("Great job reviewing the content!").classes("text-base text-green-800")
                                                                ui.separator().classes("bg-green-300 w-24 my-2")
                                                                ui.label(f"You answered {len(quiz_state['questions'])} questions").classes("text-sm text-green-700")
                                                                ui.button("Start New Quiz", on_click=lambda: (
                                                                    quiz_state.update({"quiz_started": False, "current_index": 0, "questions": []}),
                                                                    show_quiz_setup()
                                                                ), icon="refresh").props("unelevated no-caps").classes("shadow-md mt-3").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); border-radius: 8px; padding: 10px 20px; font-size: 0.875rem;")
                                                    return
                                                
                                                q = quiz_state["questions"][quiz_state["current_index"]]
                                                
                                                with quiz_content_container:
                                                    # Progress indicator - modern design
                                                    progress_percent = int(((quiz_state['current_index'] + 1) / len(quiz_state['questions'])) * 100)
                                                    with ui.card().classes("w-full border-0 shadow-sm mb-4").style("background: linear-gradient(135deg, rgba(255, 75, 75, 0.05) 0%, rgba(255, 107, 107, 0.02) 100%); padding: 16px;"):
                                                        with ui.row().classes("w-full items-center justify-between"):
                                                            with ui.row().classes("items-center gap-2"):
                                                                ui.icon("quiz", size="sm").classes("text-primary")
                                                                ui.label(f"Question {quiz_state['current_index'] + 1} of {len(quiz_state['questions'])}").classes("text-sm font-bold text-gray-700 dark:text-gray-300")
                                                            with ui.row().classes("items-center gap-2"):
                                                                ui.label(f"{progress_percent}%").classes("text-xs font-semibold text-primary")
                                                                ui.linear_progress(
                                                                    value=(quiz_state['current_index'] + 1) / len(quiz_state['questions']),
                                                                    show_value=False
                                                                ).props("color=primary size=8px").classes("w-32")
                                                
                                                    # Question card - modern professional style
                                                    with ui.card().classes("w-full border-0 shadow-xl").style("background: #ffffff; border: 1px solid #e6e9ef; padding: 24px;"):
                                                        with ui.column().classes("gap-4 w-full"):
                                                            # Question text
                                                            ui.label(q["question"]).classes("text-base font-semibold text-gray-900 dark:text-white mb-2")
                                                            
                                                            if not quiz_state["show_answer"]:
                                                                # Show options - matching AI Tutor suggestion button style
                                                                ui.separator().classes("my-2")
                                                                for idx, option in enumerate(q.get("options", [])):
                                                                    with ui.button(
                                                                        on_click=lambda opt=option: handle_answer(opt)
                                                                    ).props("outline no-caps").classes("w-full hover:bg-primary/5 mb-2").style("justify-content: flex-start; padding: 12px 16px; border-radius: 8px; border-color: rgba(255, 75, 75, 0.2);"):
                                                                        with ui.row().classes("items-center gap-3 w-full").style("flex-wrap: nowrap;"):
                                                                            with ui.avatar(size="sm").classes("text-xs font-bold").style(f"background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 107, 107, 0.05) 100%); color: #FF4B4B;"):
                                                                                ui.label(chr(65 + idx))  # A, B, C, D
                                                                            ui.label(option).classes("text-sm text-gray-700 dark:text-gray-300 flex-grow text-left")
                                                            else:
                                                                # Show answer and explanation - modern professional style
                                                                ui.separator().classes("my-3")
                                                                with ui.card().classes("w-full border-0 shadow-lg").style("background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-left: 4px solid #10b981; padding: 20px;"):
                                                                    with ui.column().classes("gap-3 w-full"):
                                                                        with ui.row().classes("items-center gap-2 mb-2"):
                                                                            ui.icon("check_circle", size="md").classes("text-green-700")
                                                                            ui.label("Correct Answer").classes("font-bold text-base text-green-900")
                                                                        ui.label(q["correct_answer"]).classes("text-base font-semibold text-green-800 ml-8")
                                                                        
                                                                        if q.get("explanation"):
                                                                            ui.separator().classes("bg-green-300 my-2")
                                                                            with ui.row().classes("items-start gap-2"):
                                                                                ui.icon("lightbulb", size="sm").classes("text-green-700 flex-shrink-0").style("margin-top: 2px;")
                                                                                with ui.column().classes("gap-1 flex-grow"):
                                                                                    ui.label("Explanation").classes("font-bold text-sm text-green-800")
                                                                                    ui.label(q["explanation"]).classes("text-sm text-green-700")
                                                                
                                                                # Navigation buttons - modern style
                                                                with ui.row().classes("w-full justify-end gap-2 mt-3"):
                                                                    ui.button(
                                                                        "Next Question" if quiz_state["current_index"] < len(quiz_state["questions"]) - 1 else "Finish Quiz",
                                                                        on_click=lambda: (
                                                                            quiz_state.update({"current_index": quiz_state["current_index"] + 1, "show_answer": False}),
                                                                            show_question()
                                                                        ),
                                                                        icon="arrow_forward"
                                                                    ).props("unelevated no-caps").classes("shadow-md").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); border-radius: 8px; padding: 10px 20px; font-size: 0.875rem;")
                                            except Exception as e:
                                                print(f"ERROR in show_question: {e}")
                                                import traceback
                                                traceback.print_exc()
                                        
                                        def handle_answer(selected):
                                            """Handle user's answer selection"""
                                            quiz_state["show_answer"] = True
                                            show_question()
                                        
                                        async def generate_quiz_async():
                                            """Generate quiz questions using LLM"""
                                            try:
                                                quiz_content_container.clear()
                                                
                                                with quiz_content_container:
                                                    with ui.card().classes("border-0 shadow-lg").style("width: 100% !important; height: 100%; min-height: 300px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);"):
                                                        with ui.column().classes("items-center justify-center gap-3"):
                                                            ui.spinner(size="lg", color="primary")
                                                            ui.label("Generating quiz questions...").classes("text-sm font-semibold text-gray-700 dark:text-gray-300")
                                                            ui.label("This may take a moment").classes("text-xs text-gray-500 dark:text-gray-400")
                                                
                                                # Build context from TCM
                                                context_parts = [f"Video Topic: {topic}\n"]
                                                if tcm_data:
                                                    context_parts.append("Video Content Summary:")
                                                    for entry in tcm_data[:10]:  # Use first 10 entries for context
                                                        concept = entry.get("conceptName", "")
                                                        narration = entry.get("narrationText", "")
                                                        if concept and narration:
                                                            context_parts.append(f"- {concept}: {narration[:200]}")
                                                
                                                context = "\n".join(context_parts)
                                                
                                                prompt = f"""Based on this educational video content, generate {quiz_settings["num_questions"]} {quiz_settings["question_type"]} quiz questions at {quiz_settings["difficulty"]} difficulty level.

{context}

Generate questions in this EXACT JSON format:
[
{{
    "question": "Question text here?",
    "type": "multiple_choice",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Brief explanation of why this is correct"
}}
]

For True/False questions, use options: ["True", "False"]
Make questions relevant to the video content and educational."""

                                                quiz_llm = LiteLLMWrapper(model_name=app_state["planner_model_name"])
                                                response = await asyncio.get_event_loop().run_in_executor(
                                                    None,
                                                    lambda: quiz_llm([{"type": "text", "content": prompt}])
                                                )
                                                
                                                # Parse JSON response
                                                import json
                                                import re
                                                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                                                if json_match:
                                                    quiz_state["questions"] = json.loads(json_match.group())
                                                    quiz_state["current_index"] = 0
                                                    quiz_state["show_answer"] = False
                                                    quiz_state["quiz_started"] = True
                                                    show_question()
                                                else:
                                                    raise ValueError("Could not parse quiz questions")
                                                    
                                            except Exception as e:
                                                quiz_content_container.clear()
                                                with quiz_content_container:
                                                    with ui.card().classes("w-full border-0 shadow-lg").style("background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-left: 4px solid #ef4444; padding: 24px;"):
                                                        with ui.column().classes("gap-3 w-full"):
                                                            with ui.row().classes("items-center gap-2"):
                                                                ui.icon("error_outline", size="md").classes("text-red-600")
                                                                ui.label("Error Generating Quiz").classes("text-lg font-bold text-red-900")
                                                            ui.label(f"{str(e)}").classes("text-sm text-red-700 ml-8")
                                                            ui.button("Try Again", on_click=show_quiz_setup, icon="refresh").props("unelevated no-caps").classes("shadow-md mt-2").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); border-radius: 8px; padding: 10px 16px; font-size: 0.875rem;")
                                        
                                        def show_quiz_setup():
                                            """Display the initial quiz configuration form"""
                                            try:
                                                quiz_content_container.clear()
                                                with quiz_content_container:
                                                    # Initial quiz setup - matching AI Tutor welcome card
                                                    with ui.card().classes("w-full border-0 shadow-lg").style("background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%); padding: 28px; border-left: 4px solid #FF4B4B;"):
                                                        # Instructions - Row 1
                                                        with ui.row().classes("items-center gap-2 mb-3"):
                                                            ui.icon("info_outline", size="sm").classes("text-blue-500 flex-shrink-0")
                                                            ui.label("Configure your quiz settings below and click Generate Quiz to start:").classes("text-xs text-gray-700 dark:text-gray-300 flex-grow")
                                                        
                                                        # Settings - Row 2
                                                        with ui.row().classes("gap-3 mb-3").style("width: 100% !important; display: flex; box-sizing: border-box;"):
                                                            with ui.column().style("flex: 1; min-width: 0;"):
                                                                with ui.row().classes("items-center gap-2 mb-1"):
                                                                    ui.icon("numbers", size="sm").classes("text-primary")
                                                                    ui.label("Questions").classes("text-sm font-semibold text-gray-700 dark:text-gray-300")
                                                                num_questions = ui.number(value=quiz_settings["num_questions"], min=1, max=10, step=1, on_change=lambda e: quiz_settings.update({"num_questions": int(e.value)})).props("outlined dense").style("width: 100%; border-radius: 8px;")
                                                            
                                                            with ui.column().style("flex: 1; min-width: 0;"):
                                                                with ui.row().classes("items-center gap-2 mb-1"):
                                                                    ui.icon("category", size="sm").classes("text-primary")
                                                                    ui.label("Type").classes("text-sm font-semibold text-gray-700 dark:text-gray-300")
                                                                question_type = ui.select(
                                                                    options=["Multiple Choice", "True/False", "Mixed"],
                                                                    value=quiz_settings["question_type"],
                                                                    on_change=lambda e: quiz_settings.update({"question_type": e.value})
                                                                ).props("outlined dense").style("width: 100%; border-radius: 8px;")
                                                            
                                                            with ui.column().style("flex: 1; min-width: 0;"):
                                                                with ui.row().classes("items-center gap-2 mb-1"):
                                                                    ui.icon("speed", size="sm").classes("text-primary")
                                                                    ui.label("Difficulty").classes("text-sm font-semibold text-gray-700 dark:text-gray-300")
                                                                difficulty = ui.select(
                                                                    options=["Easy", "Medium", "Hard"],
                                                                    value=quiz_settings["difficulty"],
                                                                    on_change=lambda e: quiz_settings.update({"difficulty": e.value})
                                                                ).props("outlined dense").style("width: 100%; border-radius: 8px;")
                                                    
                                                        # Separator
                                                        ui.separator().classes("my-2")
                                                        
                                                        # Generate Button
                                                        ui.button(
                                                            "Generate Quiz",
                                                            on_click=generate_quiz_async,
                                                            icon="auto_awesome"
                                                        ).props("unelevated no-caps").classes("shadow-md mt-2 w-full").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); border-radius: 8px; padding: 10px 16px; font-size: 0.875rem;")
                                                
                                            except Exception as e:
                                                print(f"ERROR in show_quiz_setup: {e}")
                                                import traceback
                                                traceback.print_exc()
                                        
                                        # Show initial setup
                                        show_quiz_setup()
                            
                            with ui.column().classes("lg:col-span-2"):
                                # Add Streamlit-style chat CSS
                                ui.add_head_html("""
                                <style>
                                    /* Streamlit-inspired chat message styling */
                                    .stchat-message {
                                        display: flex;
                                        gap: 16px;
                                        margin-bottom: 1rem;
                                        padding: 1rem;
                                        border-radius: 0.5rem;
                                        animation: fadeIn 0.3s ease-in;
                                        align-items: flex-start;
                                        width: 100% !important;
                                        max-width: 100% !important;
                                        box-sizing: border-box !important;
                                    }
                                    
                                    @keyframes fadeIn {
                                        from { opacity: 0; transform: translateY(10px); }
                                        to { opacity: 1; transform: translateY(0); }
                                    }
                                    
                                    /* Ensure chat container children don't overflow */
                                    #chat_container {
                                        overflow-x: hidden !important;
                                        width: 100% !important;
                                    }
                                    
                                    #chat_container > * {
                                        max-width: 100% !important;
                                        box-sizing: border-box !important;
                                    }
                                    
                                    /* User message - light background with avatar on right */
                                    .stchat-message.user {
                                        background-color: #f0f2f6;
                                        flex-direction: row-reverse;
                                    }
                                    
                                    body.dark .stchat-message.user {
                                        background-color: #262730;
                                    }
                                    
                                    /* AI message - slightly different background */
                                    .stchat-message.assistant {
                                        background-color: #ffffff;
                                        border: 1px solid #e6e9ef;
                                    }
                                    
                                    body.dark .stchat-message.assistant {
                                        background-color: #1a1d24;
                                        border-color: #262c36;
                                    }
                                    
                                    /* Avatar styling */
                                    .stchat-avatar {
                                        width: 40px;
                                        height: 40px;
                                        border-radius: 0.5rem;
                                        flex-shrink: 0;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-weight: 600;
                                        font-size: 18px;
                                    }
                                    
                                    .stchat-message.user .stchat-avatar {
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        color: white;
                                    }
                                    
                                    .stchat-message.assistant .stchat-avatar {
                                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                        color: white;
                                    }
                                    
                                    /* Message content */
                                    .stchat-content {
                                        flex: 1 1 0;
                                        min-width: 0 !important;
                                        max-width: calc(100% - 56px) !important;
                                        color: #31333F;
                                        line-height: 1.6;
                                        overflow-wrap: break-word !important;
                                        word-wrap: break-word !important;
                                    }
                                    
                                    .stchat-content * {
                                        max-width: 100% !important;
                                        box-sizing: border-box !important;
                                        overflow-wrap: break-word !important;
                                        word-wrap: break-word !important;
                                    }
                                    
                                    body.dark .stchat-content {
                                        color: #fafafa;
                                    }
                                    
                                    .stchat-content p {
                                        margin: 0 0 0.5rem 0 !important;
                                        overflow-wrap: break-word !important;
                                        word-wrap: break-word !important;
                                        word-break: break-word !important;
                                        white-space: pre-wrap !important;
                                    }
                                    
                                    .stchat-content p:last-child {
                                        margin-bottom: 0 !important;
                                    }
                                    
                                    .stchat-content a {
                                        word-break: break-all !important;
                                        overflow-wrap: anywhere !important;
                                    }
                                    
                                    /* Force all text content to wrap */
                                    .stchat-content span,
                                    .stchat-content div {
                                        overflow-wrap: break-word !important;
                                        word-wrap: break-word !important;
                                        word-break: break-word !important;
                                    }
                                    
                                    .stchat-content code {
                                        background-color: rgba(151, 166, 195, 0.15);
                                        padding: 0.2em 0.4em;
                                        border-radius: 0.25rem;
                                        font-size: 0.875em;
                                        font-family: 'Source Code Pro', monospace;
                                        word-break: break-all;
                                        overflow-wrap: anywhere;
                                    }
                                    
                                    body.dark .stchat-content code {
                                        background-color: rgba(250, 250, 250, 0.1);
                                    }
                                    
                                    .stchat-content pre {
                                        background-color: #f0f2f6;
                                        padding: 1rem;
                                        border-radius: 0.5rem;
                                        overflow-x: auto;
                                        margin: 0.5rem 0;
                                        max-width: 100%;
                                        white-space: pre-wrap;
                                        word-wrap: break-word;
                                    }
                                    
                                    body.dark .stchat-content pre {
                                        background-color: #262730;
                                    }
                                    
                                    .stchat-content pre code {
                                        background: none;
                                        padding: 0;
                                    }
                                    
                                    /* Strong/bold text */
                                    .stchat-content strong {
                                        font-weight: 600;
                                        color: #FF4B4B;
                                    }
                                    
                                    body.dark .stchat-content strong {
                                        color: #FF6B6B;
                                    }
                                </style>
                                """)
                                
                                with ui.card().classes("w-full shadow-2xl border-0").style("height: calc(100vh - 200px); min-height: 500px; max-height: 700px; display: flex; flex-direction: column; overflow: hidden;"):
                                    
                                    # ---------------- HEADER ----------------
                                    with ui.row().classes(
                                        "w-full items-center justify-between px-5 py-3 border-b dark:border-gray-700"
                                    ).style("flex-shrink: 0; background: linear-gradient(135deg, rgba(255, 75, 75, 0.05) 0%, rgba(255, 107, 107, 0.02) 100%);"):
                                        with ui.row().classes("items-center gap-3"):
                                            with ui.avatar(size="md").classes("shadow-md").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);"):
                                                ui.icon("psychology", size="sm").classes("text-white")
                                            with ui.column().classes("gap-0"):
                                                ui.label("AI Tutor").classes(
                                                    "text-lg font-bold text-gray-900 dark:text-white"
                                                )
                                                ui.label("Your intelligent learning companion").classes(
                                                    "text-xs text-gray-500 dark:text-gray-400"
                                                )
                                        
                                        def clear_chat():
                                            chat_history.clear()
                                            chat_container.clear()
                                            
                                            # Recreate the welcome card with suggestions
                                            with chat_container:
                                                with ui.card().classes(
                                                    "w-full border-0 shadow-lg"
                                                ).style("background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%); padding: 28px; border-left: 4px solid #FF4B4B;"):
                                                    # Instructions
                                                    with ui.row().classes("items-start gap-2 mb-3").style("flex-wrap: nowrap;"):
                                                        ui.icon("info_outline", size="sm").classes("text-blue-500 flex-shrink-0").style("margin-top: 2px;")
                                                        ui.label(
                                                            "Pause the video at any moment to ask questions. Try these suggestions:"
                                                        ).classes("text-xs text-gray-700 dark:text-gray-300 flex-grow")
                                                    
                                                    # Suggestion buttons
                                                    suggestions = [
                                                        ("help_outline", "Explain this in simpler terms"),
                                                        ("lightbulb", "Give me a real-world example"),
                                                        ("list", "Break down this step-by-step"),
                                                        ("star", "Why is this concept important?"),
                                                    ]
                                                    
                                                    for icon, suggestion in suggestions:
                                                        with ui.button(
                                                            on_click=lambda s=suggestion: (
                                                                setattr(chat_input, 'value', s),
                                                                chat_input.run_method("focus")
                                                            )
                                                        ).props("outline no-caps").classes(
                                                            "w-full mb-2 hover:bg-primary/5"
                                                        ).style("justify-content: flex-start; padding: 12px 16px; border-radius: 8px; border-color: rgba(255, 75, 75, 0.2);"):
                                                            with ui.row().classes("items-center gap-3").style("flex-wrap: nowrap;"):
                                                                ui.icon(icon, size="sm").classes("text-primary flex-shrink-0")
                                                                ui.label(suggestion).classes("text-sm text-gray-700 dark:text-gray-300 flex-grow")
                                            
                                            ui.notify("Chat history cleared", type="positive", icon="check_circle")
                                        
                                        ui.button(
                                            icon="refresh",
                                            on_click=clear_chat
                                        ).props("flat round dense").classes("text-gray-600 dark:text-gray-400 hover:text-primary hover:bg-primary/10").tooltip("Clear chat history")

                                    # ---------------- CHAT CONTAINER ----------------
                                    with ui.column().style("flex: 1; min-height: 0; position: relative; overflow: hidden;"):
                                        chat_container = ui.column().classes(
                                            "h-full p-4 overflow-y-auto overflow-x-hidden gap-3"
                                        ).props('id="chat_container"').style("scroll-behavior: smooth;")
                                        
                                        # Scroll to bottom button (initially hidden)
                                        scroll_btn = ui.button(
                                            icon="arrow_downward",
                                            on_click=lambda: ui.run_javascript(
                                                'document.getElementById("chat_container").scrollTop = '
                                                'document.getElementById("chat_container").scrollHeight'
                                            )
                                        ).props("fab-mini color=primary").classes(
                                            "absolute bottom-4 right-4 shadow-lg"
                                        ).style("display: none;")
                                        scroll_btn.tooltip("Scroll to bottom")
                                        
                                        # Show/hide scroll button based on scroll position
                                        ui.add_body_html("""
                                            <script>
                                            setTimeout(() => {
                                                const container = document.getElementById('chat_container');
                                                if (container) {
                                                    container.addEventListener('scroll', () => {
                                                        const scrollBtn = container.parentElement.querySelector('.q-btn');
                                                        if (scrollBtn) {
                                                            const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
                                                            scrollBtn.style.display = isNearBottom ? 'none' : 'block';
                                                        }
                                                    });
                                                }
                                            }, 500);
                                            </script>
                                        """)

                                    # ------------------------------------------------------
                                    # LOAD HISTORY
                                    # ------------------------------------------------------
                                    chat_history = app_state["chat_histories"].setdefault(
                                        topic, []
                                    )

                                    with chat_container:
                                        if not chat_history:
                                            # Professional welcome card
                                            with ui.card().classes(
                                                "w-full border-0 shadow-lg"
                                            ).style("background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%); padding: 28px; border-left: 4px solid #FF4B4B;"):
                                                # Instructions
                                                with ui.row().classes("items-start gap-2 mb-3").style("flex-wrap: nowrap;"):
                                                    ui.icon("info_outline", size="sm").classes("text-blue-500 flex-shrink-0").style("margin-top: 2px;")
                                                    ui.label(
                                                        "Pause the video at any moment to ask questions. Try these suggestions:"
                                                    ).classes("text-xs text-gray-700 dark:text-gray-300 flex-grow")
                                                
                                                # Suggestion buttons
                                                suggestions = [
                                                    ("help_outline", "Explain this in simpler terms"),
                                                    ("lightbulb", "Give me a real-world example"),
                                                    ("list", "Break down this step-by-step"),
                                                    ("star", "Why is this concept important?"),
                                                ]
                                                
                                                for icon, suggestion in suggestions:
                                                    with ui.button(
                                                        on_click=lambda s=suggestion: (
                                                            setattr(chat_input, 'value', s),
                                                            chat_input.run_method("focus")
                                                        )
                                                    ).props("outline no-caps").classes(
                                                        "w-full mb-2 hover:bg-primary/5"
                                                    ).style("justify-content: flex-start; padding: 12px 16px; border-radius: 8px; border-color: rgba(255, 75, 75, 0.2);"):
                                                        with ui.row().classes("items-center gap-3").style("flex-wrap: nowrap;"):
                                                            ui.icon(icon, size="sm").classes("text-primary flex-shrink-0")
                                                            ui.label(suggestion).classes("text-sm text-gray-700 dark:text-gray-300 flex-grow")
                                        else:
                                            # Load existing messages with custom styling
                                            for msg in chat_history:
                                                is_user = msg["role"] == "user"
                                                render_chat_message(chat_container, msg["content"], is_user)

                                    # ------------------------------------------------------
                                    # INPUT BAR
                                    # ------------------------------------------------------
                                    with ui.column().classes("w-full").style("flex-shrink: 0; background: linear-gradient(to top, rgba(255,255,255,0.98), rgba(255,255,255,0.95)); backdrop-filter: blur(10px); border-top: 1px solid rgba(0,0,0,0.08); padding-top: 16px;"):
                                        # Input row
                                        with ui.row().classes("w-full items-end gap-3 px-5 pb-3"):
                                            with ui.card().classes("flex-grow shadow-sm border-0").style("background: white; padding: 8px 16px; border-radius: 20px; border: 2px solid rgba(255, 75, 75, 0.15); overflow: hidden;"):
                                                chat_input = (
                                                    ui.textarea(placeholder="Ask me anything about the video...")
                                                    .props("borderless dense autogrow")
                                                    .classes("w-full")
                                                    .style("max-height: 100px; min-height: 40px; font-size: 14px; overflow-y: auto; word-wrap: break-word; overflow-wrap: break-word;")
                                                )
                                            
                                            # Send button
                                            send_button = ui.button(
                                                icon="send"
                                            ).props("round unelevated").classes("shadow-lg").style(
                                                "flex-shrink: 0; width: 48px; height: 48px; background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);"
                                            )
                                            send_button.tooltip("Send message")
                                    
                                    # JavaScript to handle Enter key properly
                                    ui.add_body_html("""
                                    <script>
                                    setTimeout(() => {
                                        const textarea = document.querySelector('textarea[placeholder="Ask me anything about the video..."]');
                                        if (textarea) {
                                            textarea.addEventListener('keydown', (e) => {
                                                if (e.key === 'Enter' && !e.shiftKey) {
                                                    e.preventDefault();
                                                    // Find the send button (the round button with send icon)
                                                    const container = textarea.closest('.q-page');
                                                    const sendBtn = container.querySelector('button[aria-label*="Send"]') || 
                                                                   container.querySelector('.q-btn--round[style*="gradient"]');
                                                    if (sendBtn) {
                                                        sendBtn.click();
                                                    }
                                                }
                                            });
                                        }
                                    }, 800);
                                    </script>
                                    """)
                                    
                                    # Auto-focus input on load
                                    ui.timer(0.1, lambda: chat_input.run_method("focus"), once=True)

                                    # ------------------------------------------------------
                                    # SEND MESSAGE FUNCTION (define before use)
                                    # ------------------------------------------------------
                                    async def send_message():
                                        # Retrieve text from input box
                                        question = chat_input.value or ""
                                        
                                        if not question.strip():
                                            return
                                        
                                        # Clear input and disable controls
                                        chat_input.value = ""
                                        chat_input.disable()
                                        send_button.disable()

                                        try:
                                            # Store in history
                                            chat_history.append(
                                                {"role": "user", "content": question}
                                            )

                                            # Display user message
                                            render_chat_message(chat_container, question, is_user=True)

                                            # Auto-scroll to bottom
                                            await ui.run_javascript(
                                            'document.getElementById("chat_container").scrollTop = '
                                            'document.getElementById("chat_container").scrollHeight'
                                            )

                                            # Show Streamlit-style typing indicator
                                            with chat_container:
                                                with ui.element('div').classes("stchat-message assistant") as typing_indicator:
                                                    with ui.element('div').classes("stchat-avatar"):
                                                        ui.label("🤖").classes("text-xl")
                                                    with ui.element('div').classes("stchat-content"):
                                                        with ui.row().classes("items-center gap-2"):
                                                            ui.spinner(size="sm", color="primary")
                                                            ui.label("Thinking...").classes("text-sm")
                                    
                                            # Auto-scroll to show typing indicator
                                            await ui.run_javascript(
                                            'document.getElementById("chat_container").scrollTop = '
                                            'document.getElementById("chat_container").scrollHeight'
                                            )

                                            # Build LLM prompt with rich TCM context
                                            current_entry = app_state["current_tcm_entry"] or {}
                                            timestamp = app_state.get("latest_pause_time", 0.0)
                                    
                                            # Load system prompt from file
                                            try:
                                                with open("ai_tutor_system_prompt.txt", "r", encoding="utf-8") as f:
                                                    system_prompt_template = f.read()
                                                # Replace placeholders
                                                system_prompt = system_prompt_template.format(
                                                    topic=topic,
                                                    time=f"{timestamp:.1f}"
                                                )
                                            except FileNotFoundError:
                                                # Fallback if file doesn't exist
                                                system_prompt = (
                                                f"You are an AI tutor helping students understand educational video content about {topic}. "
                                                f"The student paused at {timestamp:.1f}s. "
                                                "Provide clear, concise explanations based on the context provided."
                                            )
                                    
                                            # Build rich context from TCM
                                            context_parts = [f"=== VIDEO: {topic} ===\n"]
                                    
                                            if current_entry:
                                                concept = current_entry.get("conceptName", "Unknown")
                                                narration = current_entry.get("narrationText", "")
                                                visual = current_entry.get("visualDescription", "")
                                                concept_id = current_entry.get("conceptId", "")
                                            
                                                context_parts.append(f"STUDENT PAUSED AT: {timestamp:.1f} seconds")
                                                context_parts.append(f"\n--- WHAT'S HAPPENING RIGHT NOW ---")
                                                context_parts.append(f"Concept Being Explained: {concept}")
                                                if narration:
                                                    context_parts.append(f"\nNarration (what's being said):\n\"{narration}\"")
                                                if visual:
                                                    context_parts.append(f"\nVisual (what's on screen):\n{visual}")
                                                if concept_id:
                                                    context_parts.append(f"\nConcept ID: {concept_id}")
                                            else:
                                                context_parts.append("(Video is currently playing - student needs to pause to get specific context)")
                                        
                                            # Add surrounding context for better understanding
                                            current_idx = next((i for i, e in enumerate(tcm_data) if e == current_entry), -1)
                                        
                                            if current_idx > 0:
                                                prev_entry = tcm_data[current_idx - 1]
                                                context_parts.append(f"\n--- WHAT CAME BEFORE ---")
                                                context_parts.append(f"Previous Concept: {prev_entry.get('conceptName', 'Unknown')}")
                                                prev_narration = prev_entry.get('narrationText', '')
                                                if len(prev_narration) > 150:
                                                    prev_narration = prev_narration[:150] + "..."
                                                context_parts.append(f"Previous Narration: \"{prev_narration}\"")
                                        
                                            if current_idx < len(tcm_data) - 1:
                                                next_entry = tcm_data[current_idx + 1]
                                                context_parts.append(f"\n--- WHAT COMES NEXT ---")
                                                context_parts.append(f"Next Concept: {next_entry.get('conceptName', 'Unknown')}")
                                                next_narration = next_entry.get('narrationText', '')
                                                if len(next_narration) > 150:
                                                    next_narration = next_narration[:150] + "..."
                                                context_parts.append(f"Next Narration: \"{next_narration}\"")
                                            else:
                                                context_parts.append("(Video is currently playing - student needs to pause to get specific context)")
                                    
                                            context_info = "\n".join(context_parts)
                                    
                                            # Build messages in the format expected by LiteLLMWrapper
                                            messages_for_llm = [
                                            {"type": "system", "content": system_prompt},
                                            {"type": "text", "content": f"VIDEO CONTEXT:\n{context_info}"},
                                            {"type": "text", "content": f"STUDENT QUESTION: {question}"}
                                            ]

                                            # Call model wrapper
                                            chat_llm = LiteLLMWrapper(
                                            model_name=app_state["planner_model_name"]
                                            )
                                            # Run in executor to avoid blocking
                                            loop = asyncio.get_event_loop()
                                            response = await loop.run_in_executor(
                                            None,
                                            lambda: chat_llm(messages_for_llm)
                                            )

                                            # Remove typing indicator
                                            chat_container.remove(typing_indicator)

                                            # Add assistant response to history
                                            chat_history.append(
                                            {"role": "assistant", "content": response}
                                            )

                                            # Display assistant message
                                            render_chat_message(chat_container, response, is_user=False)

                                            # Auto-scroll to bottom
                                            await ui.run_javascript(
                                            'document.getElementById("chat_container").scrollTop = '
                                            'document.getElementById("chat_container").scrollHeight'
                                            )
                                        
                                        except Exception as e:
                                            # Remove typing indicator if it exists
                                            try:
                                                chat_container.remove(typing_indicator)
                                            except:
                                                pass
                                        
                                            # Show error message
                                            error_msg = str(e) if str(e) else repr(e)
                                            with chat_container:
                                                with ui.element('div').classes("stchat-message assistant").style("background-color: #fee2e2; border-color: #fca5a5;"):
                                                    with ui.element('div').classes("stchat-avatar").style("background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);"):
                                                        ui.label("⚠️").classes("text-xl")
                                                    with ui.element('div').classes("stchat-content").style("color: #991b1b;"):
                                                        ui.markdown(f"**Error:** Unable to get response.\n\n`{error_msg}`")
                                        
                                            ui.notify(
                                                f"Failed to get AI response: {error_msg}", 
                                                type="negative"
                                            )
                                        
                                        finally:
                                            # Re-enable controls
                                            chat_input.enable()
                                            send_button.enable()
                                            chat_input.run_method("focus")
                                    
                                    # Connect button to function
                                    send_button.on_click(send_message)

                                # ------------------------------------------------------
                                # UPDATE VIDEO CONTEXT
                                # ------------------------------------------------------
                                async def update_context(
                                    e: events.GenericEventArguments,
                                ):
                                    ts = await ui.run_javascript(
                                        "document.getElementById('video_player')?.currentTime || 0"
                                    )

                                    try:
                                        ts = float(ts)
                                    except Exception:
                                        ts = 0.0

                                    app_state["latest_pause_time"] = ts

                                    def valid_entry(entry):
                                        try:
                                            return (
                                                float(entry["startTime"])
                                                <= ts
                                                < float(entry["endTime"])
                                            )
                                        except Exception:
                                            return False

                                    found_entry = next(
                                        (x for x in tcm_data if valid_entry(x)), None
                                    )
                                    app_state["current_tcm_entry"] = found_entry

                                    if found_entry:
                                        concept_name = found_entry.get('conceptName', 'Unknown')
                                        narration_preview = found_entry.get('narrationText', '')[:80]
                                        if len(found_entry.get('narrationText', '')) > 80:
                                            narration_preview += "..."
                                        
                                        context_label.set_content(
                                            f"⏸️ **Current Topic:** {concept_name} (`{ts:.1f}s`)\n\n"
                                            f"_{narration_preview}_"
                                        )
                                        context_label.classes(
                                            "p-3 text-sm bg-green-50 dark:bg-green-900/20 "
                                            "border-l-4 border-green-500 rounded-r",
                                            remove="bg-blue-50 dark:bg-gray-800 border-blue-500"
                                        )
                                    else:
                                        context_label.set_content(
                                            "▶️ **Pause the video** to ask a question about what you're watching."
                                        )
                                        context_label.classes(
                                            "p-3 text-sm bg-blue-50 dark:bg-gray-800 "
                                            "border-l-4 border-blue-500 rounded-r",
                                            remove="bg-green-50 dark:bg-green-900/20 border-green-500"
                                        )

                                video_player.on("seeked", update_context)
                                video_player.on("pause", update_context)
                                video_player.on("play", lambda e: context_label.set_content(
                                    "▶️ **Video Playing** - Pause to ask questions."
                                ))

                    else:
                        # Better empty state for missing video
                        with ui.card().classes("w-full p-12 text-center"):
                            ui.icon("video_camera_back", size="4rem").classes("text-gray-300 dark:text-gray-600 mb-4")
                            ui.label("Video Not Generated Yet").classes("text-xl font-semibold text-gray-600 dark:text-gray-400 mb-2")
                            ui.label(
                                "Use the Utilities tab to continue and complete the video generation for this project."
                            ).classes("text-gray-500 dark:text-gray-500 mb-4")
                            ui.button("Go to Utilities", icon="build", on_click=lambda: main_tabs.set_value("🔧 Utilities")).props("unelevated no-caps").classes("mt-2")

                with ui.tab_panel(t_plan):
                    outline_path = os.path.join(
                        project_path, f"{inner_folder_name}_scene_outline.txt"
                    )
                    if os.path.exists(outline_path):
                        # Parse the scene outline
                        outline_content = safe_read_file(outline_path, clean=False)
                        
                        # Extract scenes using regex
                        import re
                        scene_pattern = r'<SCENE_(\d+)>(.*?)</SCENE_\1>'
                        scenes = re.findall(scene_pattern, outline_content, re.DOTALL)
                        
                        if scenes:
                            # Header card
                            with ui.card().classes("w-full shadow-lg border-l-4 border-primary mb-6"):
                                with ui.row().classes("w-full items-center gap-3 p-6"):
                                    ui.icon("movie_creation", size="lg").classes("text-primary")
                                    with ui.column().classes("gap-1"):
                                        ui.label(topic).classes("text-2xl font-bold text-gray-900 dark:text-white")
                                        ui.label(f"{len(scenes)} scenes planned").classes("text-sm text-gray-600 dark:text-gray-400")
                            
                            # Scene cards
                            for scene_num, scene_content in scenes:
                                # Extract scene details
                                title_match = re.search(r'Scene Title:\s*(.+?)(?:\n|$)', scene_content)
                                purpose_match = re.search(r'Scene Purpose:\s*(.+?)(?:\n|Scene Description)', scene_content, re.DOTALL)
                                desc_match = re.search(r'Scene Description:\s*(.+?)(?:\n|Scene Layout)', scene_content, re.DOTALL)
                                
                                title = title_match.group(1).strip() if title_match else f"Scene {scene_num}"
                                purpose = purpose_match.group(1).strip() if purpose_match else ""
                                description = desc_match.group(1).strip() if desc_match else ""
                                
                                with ui.card().classes("w-full shadow-md hover:shadow-lg transition-shadow mb-4"):
                                    # Scene header
                                    with ui.row().classes("w-full items-start gap-3 p-4 bg-gradient-to-r from-primary/5 to-transparent border-b dark:border-gray-700"):
                                        with ui.column().classes("gap-2 flex-grow"):
                                            # Title with scene number
                                            with ui.row().classes("items-center gap-2"):
                                                ui.label(f"Scene {scene_num}:").classes("text-lg font-bold text-primary")
                                                ui.label(title).classes("text-lg font-semibold text-gray-900 dark:text-white")
                                            
                                            # Purpose with inline icon
                                            if purpose:
                                                with ui.row().classes("items-center gap-1.5").style("flex-wrap: nowrap;"):
                                                    ui.icon("lightbulb", size="xs").classes("text-amber-500 flex-shrink-0")
                                                    ui.label(purpose).classes("text-sm text-gray-600 dark:text-gray-400")
                                    
                                    # Scene content with inline icon
                                    if description:
                                        with ui.row().classes("items-start gap-2 p-4").style("flex-wrap: nowrap;"):
                                            ui.icon("description", size="sm").classes("text-blue-500 flex-shrink-0").style("margin-top: 2px;")
                                            ui.label(description).classes("text-sm text-gray-700 dark:text-gray-300 leading-relaxed flex-grow")
                        else:
                            # Fallback to showing formatted text if parsing fails
                            with ui.card().classes("w-full shadow-lg"):
                                with ui.row().classes("w-full items-center gap-2 p-4 border-b dark:border-gray-700 bg-gray-50 dark:bg-gray-800"):
                                    ui.icon("description", size="sm").classes("text-primary")
                                    ui.label("Video Plan").classes("text-lg font-semibold")
                                with ui.column().classes("p-6"):
                                    ui.markdown(outline_content).classes("prose dark:prose-invert max-w-none")
                    else:
                        with ui.card().classes("w-full p-12 text-center"):
                            ui.icon("description", size="4rem").classes("text-gray-300 dark:text-gray-600 mb-4")
                            ui.label("Plan Not Available").classes("text-xl font-semibold text-gray-600 dark:text-gray-400")
                            ui.label("The video plan hasn't been generated yet").classes("text-sm text-gray-500 dark:text-gray-500 mt-2")

                for i, scene_dir_name in enumerate(scene_dirs):
                    with ui.tab_panel(scene_tabs_list[i]):
                        scene_num = i + 1
                        scene_path = os.path.join(project_path, scene_dir_name)
                        
                        # Scene header
                        with ui.card().classes("w-full mb-4 shadow-md border-l-4 border-secondary"):
                            with ui.row().classes("w-full items-center p-3"):
                                ui.icon("movie_filter", size="md").classes("text-secondary")
                                ui.label(f"Scene {scene_num}").classes("text-xl font-bold ml-2")
                                ui.space()

                        with ui.tabs().props("dense align=left").classes("w-full") as asset_tabs:
                            at_vision = ui.tab("🗺️ Vision").props("no-caps")
                            at_tech = ui.tab("🛠️ Technical").props("no-caps")
                            at_narr = ui.tab("🎞️ Animation").props("no-caps")
                            at_code = ui.tab("💻 Code").props("no-caps")
                            at_vid = ui.tab("🎬 Output").props("no-caps")

                        with ui.tab_panels(asset_tabs, value=at_vision).classes(
                            "w-full bg-transparent mt-4"
                        ):

                            def display_plan(file_pattern, default_text, icon_name="description"):
                                files = glob.glob(
                                    os.path.join(scene_path, file_pattern)
                                )
                                if files:
                                    plan_text = safe_read_file(
                                        files[0], clean=True
                                    )  # remove XML tags etc.
                                    if not plan_text.strip():
                                        with ui.card().classes("w-full p-8 text-center"):
                                            ui.icon(icon_name, size="3rem").classes("text-gray-300 dark:text-gray-600 mb-2")
                                            ui.label(f"{default_text} is empty").classes("text-gray-500")
                                        return

                                    # Optional cleanup for readability
                                    plan_text = re.sub(
                                        r"#+\s*Scene.*", "", plan_text
                                    ).strip()
                                    plan_text = plan_text.replace(
                                        "[SCENE_VISION]", "### 🎬 Vision"
                                    )
                                    plan_text = plan_text.replace(
                                        "[STORYBOARD]", "### 🗺️ Storyboard"
                                    )
                                    plan_text = plan_text.replace(
                                        "[ANIMATION_STRATEGY]",
                                        "### 🎨 Animation Strategy",
                                    )
                                    plan_text = plan_text.replace(
                                        "[NARRATION]", "### 🎞️ Animation"
                                    )

                                    with ui.card().classes("w-full shadow-md"):
                                        with ui.column().classes("p-6"):
                                            ui.markdown(plan_text).classes("prose dark:prose-invert max-w-none")
                                else:
                                    with ui.card().classes("w-full p-8 text-center"):
                                        ui.icon(icon_name, size="3rem").classes("text-gray-300 dark:text-gray-600 mb-2")
                                        ui.label(f"{default_text} not found").classes("text-gray-500")

                            with ui.tab_panel(at_vision):
                                display_plan(
                                    "subplans/*_vision_storyboard_plan.txt",
                                    "Vision Plan",
                                    "visibility"
                                )
                            with ui.tab_panel(at_tech):
                                display_plan(
                                    "subplans/*_technical_implementation_plan.txt",
                                    "Technical Plan",
                                    "engineering"
                                )
                            with ui.tab_panel(at_narr):
                                display_plan(
                                    "subplans/*_animation_narration_plan.txt",
                                    "Narration Plan",
                                    "mic"
                                )

                            with ui.tab_panel(at_code):
                                code_path = os.path.join(scene_path, "code")
                                if os.path.exists(code_path):
                                    code_files = sorted(
                                        glob.glob(os.path.join(code_path, "*.py"))
                                    )
                                    if not code_files:
                                        with ui.card().classes("w-full p-8 text-center"):
                                            ui.icon("code", size="3rem").classes("text-gray-300 dark:text-gray-600 mb-2")
                                            ui.label("No code files found").classes("text-gray-500")
                                    else:
                                        with ui.card().classes("w-full shadow-md"):
                                            code_versions = {
                                                os.path.basename(f): f
                                                for f in code_files
                                            }
                                            
                                            # Header with file selector
                                            with ui.row().classes("w-full items-center gap-3 p-4 border-b dark:border-gray-700 bg-gray-50 dark:bg-gray-800"):
                                                ui.icon("code", size="sm").classes("text-primary")
                                                ui.label("Scene Code").classes("text-lg font-semibold")
                                                ui.space()
                                                
                                                code_display_area = ui.column().classes("w-full p-4")

                                                def show_code(path):
                                                    code_display_area.clear()
                                                    with code_display_area:
                                                        ui.code(
                                                            safe_read_file(
                                                                path, clean=False
                                                            )
                                                        ).classes("w-full")

                                                ui.select(
                                                    options=code_versions,
                                                    label="Version",
                                                    on_change=lambda e: show_code(e.value),
                                                ).props("outlined dense").style("min-width: 200px;")
                                            
                                            show_code(code_files[-1])
                                else:
                                    with ui.card().classes("w-full p-8 text-center"):
                                        ui.icon("code_off", size="3rem").classes("text-gray-300 dark:text-gray-600 mb-2")
                                        ui.label("Code not generated yet").classes("text-gray-500")

                            with ui.tab_panel(at_vid):
                                video_file = find_latest_video_for_scene(
                                    project_path, scene_num
                                )
                                if video_file:
                                    with ui.card().classes("w-full shadow-lg overflow-hidden p-0"):
                                        video_url = (
                                            f"/{os.path.relpath(video_file, '.')}".replace(
                                                "\\", "/"
                                            )
                                        )
                                        ui.video(video_url).classes("w-full").style("display: block;")
                                else:
                                    with ui.card().classes("w-full p-12 text-center"):
                                        ui.icon("videocam_off", size="4rem").classes("text-gray-300 dark:text-gray-600 mb-4")
                                        ui.label("Scene Not Rendered Yet").classes("text-xl font-semibold text-gray-600 dark:text-gray-400 mb-2")
                                        ui.label("This scene hasn't been rendered. Generate it from the main tab.").classes("text-gray-500")

    def update_util_tab():
        util_content.clear()

        async def handle_finalize(topic):
            if not topic:
                ui.notify("Please select a project.", color="warning")
                return
            finalize_button.disable()
            finalize_button.set_text("⏳ Finalizing...")
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, combine_videos, topic)
                
                # Handle result and show appropriate notification
                if result == "success":
                    ui.notify(f"✅ Project '{topic}' finalized successfully!", color="positive")
                elif result == "already_exists":
                    ui.notify(f"ℹ️ Combined assets already exist for '{topic}'", color="info")
                elif result == "no_scenes":
                    ui.notify(f"⚠️ No rendered scenes found for '{topic}'", color="warning")
                
                project_path = get_project_path("output", topic)
                inner_folder_name = os.path.basename(project_path)
                video_local_path = os.path.join(
                    project_path, f"{inner_folder_name}_combined.mp4"
                )
                if os.path.exists(video_local_path):
                    video_url = f"/{os.path.relpath(video_local_path, '.')}".replace(
                        "\\", "/"
                    )
                    finalized_video_player.set_source(video_url)
                    finalized_video_player.style("display: block;")
                await update_dashboard()
            except Exception as e:
                ui.notify(f"Error finalizing project: {e}", color="negative")
            finally:
                finalize_button.enable()
                finalize_button.set_text("🎬 Finalize Project")

        async def handle_continue(topic):
            if not topic:
                ui.notify("Please select a project.", color="warning")
                return
            
            # Get project description from outline
            project_path = get_project_path("output", topic)
            inner_folder_name = os.path.basename(project_path)
            scene_outline_path = os.path.join(project_path, f"{inner_folder_name}_scene_outline.txt")
            
            description = f"Continue generating the unfinished scenes for {topic}"
            if os.path.exists(scene_outline_path):
                with open(scene_outline_path, "r", encoding="utf-8") as f:
                    outline_content = f.read()
                    # Try to extract description from outline if available
                    if "Description:" in outline_content:
                        desc_start = outline_content.find("Description:") + len("Description:")
                        desc_end = outline_content.find("\n\n", desc_start)
                        if desc_end > desc_start:
                            description = outline_content[desc_start:desc_end].strip()
            
            continue_button.disable()
            continue_button.set_text("⏳ Resuming...")
            progress_container_continue.style("display: block;")
            progress_bar_continue.set_visibility(True)
            progress_label_continue.set_text("🚀 Starting continuation...")
            progress_bar_continue.value = 0
            log_output_continue.clear()
            log_output_continue.push(f"🚀 Starting continuation for '{topic}'")
            log_output_continue.push(f"📋 Checking project status...")
            
            try:
                # Shared state for progress updates
                import concurrent.futures
                import threading
                import sys
                from io import StringIO
                
                generation_complete = threading.Event()
                generation_error = [None]
                current_progress = {"value": 0.1, "text": "Starting..."}
                log_buffer = []
                progress_lock = threading.Lock()
                
                def progress_callback(value, text):
                    with progress_lock:
                        current_progress["value"] = value
                        current_progress["text"] = text
                
                class LogCapture:
                    def __init__(self, original_stream, buffer_list):
                        self.original_stream = original_stream
                        self.buffer_list = buffer_list
                        self.skip_patterns = [
                            "Langfuse client is disabled",
                            "LANGFUSE_PUBLIC_KEY",
                            "No video folders found",
                            "langfuse.com/docs",
                            "See our docs:",
                            "==> Loading existing implementation plans",
                            "<== Finished loading plans",
                            "Found: 0, Missing:",
                            "Generating missing implementation plans for scenes:",
                            "Loaded existing topic session ID:",
                            "Saved topic session ID to",
                        ]
                        
                    def write(self, text):
                        self.original_stream.write(text)
                        if text.strip():
                            should_skip = any(pattern in text for pattern in self.skip_patterns)
                            if not should_skip:
                                cleaned_text = text.rstrip()
                                replacements = {
                                    "STARTING VIDEO PIPELINE FOR TOPIC:": "🚀 Continuation started for:",
                                    "[PHASE 1: SCENE OUTLINE]": "📝 Phase 1: Checking video structure",
                                    "[PHASE 1 COMPLETE]": "✅ Phase 1: Video structure ready",
                                    "[PHASE 2: IMPLEMENTATION PLANS]": "🎨 Phase 2: Planning remaining scenes",
                                    "[PHASE 2 COMPLETE]": "✅ Phase 2: All scenes planned",
                                    "[PHASE 3: CODE GENERATION & RENDERING (SCENE-BY-SCENE)]": "💻 Phase 3: Rendering remaining scenes",
                                    "[PHASE 3 COMPLETE]": "✅ Phase 3: All scenes rendered",
                                    "PIPELINE FINISHED FOR TOPIC:": "✅ Continuation complete for:",
                                    "scene outline saved": "✅ Video structure saved",
                                    "Loaded existing scene outline": "✅ Using existing video structure",
                                    "Loaded existing topic session ID": "",
                                    "Total Cost:": "💰 Cost:",
                                    "Already completed scenes:": "✅ Already rendered:",
                                    "Scenes with plans ready:": "✅ Already planned:",
                                    "Scenes to render:": "⏳ Remaining to render:",
                                    "Already rendered, skipping": "✅ Already done, skipping",
                                    "Finished generating missing plans": "✅ All scene plans ready",
                                    "==> Generating scene outline": "📝 Starting: Generating video structure",
                                    "==> Scene outline generated": "✅ Finished: Video structure generated",
                                    "==> Generating scene implementations": "🎨 Starting: Planning scene details",
                                    "==> All concurrent scene implementations generated": "✅ Finished: All scene details planned",
                                    "==> Preparing to render": "💻 Starting: Scene rendering",
                                    "Starting concurrent processing": "💻 Processing scenes concurrently",
                                    "<== All scene processing tasks completed": "✅ Finished: All scenes processed",
                                }
                                
                                for old, new in replacements.items():
                                    if old in cleaned_text:
                                        cleaned_text = cleaned_text.replace(old, new)
                                
                                import re
                                scenes_match = re.search(r'Found (\d+) scenes? in outline', cleaned_text)
                                if scenes_match:
                                    num_scenes = scenes_match.group(1)
                                    scene_word = "scene" if num_scenes == "1" else "scenes"
                                    cleaned_text = f"📊 Video has {num_scenes} {scene_word}"
                                
                                scene_prefix_match = re.search(r'\[([^\]]+) \| Scene (\d+)\]', cleaned_text)
                                if scene_prefix_match:
                                    scene_num = scene_prefix_match.group(2)
                                    cleaned_text = re.sub(r'\[([^\]]+) \| Scene (\d+)\]', f'Scene {scene_num}:', cleaned_text)
                                
                                cleaned_text = re.sub(r'\[([^\]]+)\]', '', cleaned_text).strip()
                                
                                if "======================================================" in cleaned_text or not cleaned_text.strip():
                                    return
                                
                                with progress_lock:
                                    self.buffer_list.append(cleaned_text)
                                    
                    def flush(self):
                        self.original_stream.flush()
                
                def run_generation_sync():
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    
                    try:
                        sys.stdout = LogCapture(old_stdout, log_buffer)
                        sys.stderr = LogCapture(old_stderr, log_buffer)
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        loop.run_until_complete(
                            video_generator.generate_video_pipeline(
                                topic, 
                                description, 
                                max_retries=app_state["max_retries"],
                                progress_callback=progress_callback
                            )
                        )
                        loop.close()
                    except Exception as e:
                        generation_error[0] = e
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                        generation_complete.set()
                
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                generation_future = executor.submit(run_generation_sync)
                
                heartbeat_counter = 0
                last_log_count = 0
                last_progress_text = ""
                
                while not generation_complete.is_set():
                    await asyncio.sleep(0.5)
                    heartbeat_counter += 1
                    
                    with progress_lock:
                        progress_bar_continue.value = current_progress["value"]
                        
                        # Update progress label with elapsed time
                        elapsed = heartbeat_counter // 2
                        if current_progress["text"]:
                            progress_label_continue.set_text(f"{current_progress['text']} ({elapsed}s)")
                        else:
                            progress_label_continue.set_text(f"⏳ Working... ({elapsed}s)")
                        
                        # Only push new log messages, not repeated status updates
                        if len(log_buffer) > last_log_count:
                            for log_line in log_buffer[last_log_count:]:
                                log_output_continue.push(log_line)
                            last_log_count = len(log_buffer)
                            last_progress_text = current_progress["text"]
                
                if generation_error[0]:
                    raise generation_error[0]
                
                generation_future.result()
                
                with progress_lock:
                    if len(log_buffer) > last_log_count:
                        for log_line in log_buffer[last_log_count:]:
                            log_output_continue.push(log_line)
                
                progress_label_continue.set_text("🎞️ Finalizing project...")
                progress_bar_continue.value = 0.9
                
                # Check if any scenes failed
                has_failures = len(video_generator.failed_scenes) > 0
                
                if has_failures:
                    log_output_continue.push(f"\n⚠️ Generation completed with {len(video_generator.failed_scenes)} failed scene(s). Finalizing...")
                else:
                    log_output_continue.push("\n✅ All scenes generated successfully! Finalizing...")
                
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, combine_videos, topic)
                
                progress_bar_continue.value = 1.0
                
                if has_failures:
                    progress_label_continue.set_text(f"⚠️ Project completed with {len(video_generator.failed_scenes)} failed scene(s)")
                    log_output_continue.push(f"⚠️ Project finalized with {len(video_generator.failed_scenes)} scene(s) failed. Check logs for details.")
                    ui.notify(f"⚠️ Project '{topic}' completed with failures. Check logs.", color="warning", icon="warning")
                else:
                    progress_label_continue.set_text("✅ Project complete!")
                    log_output_continue.push("✅ Project finalized successfully!")
                    if result == "success":
                        ui.notify(f"✅ Project '{topic}' completed successfully!", color="positive", icon="check_circle")
                    elif result == "already_exists":
                        ui.notify(f"ℹ️ Project '{topic}' already finalized", color="info", icon="info")
                    elif result == "no_scenes":
                        ui.notify(f"⚠️ No rendered scenes found for '{topic}'", color="warning", icon="warning")
                
                await asyncio.sleep(2)
                progress_container_continue.style("display: none;")
                await update_dashboard()
                
            except Exception as e:
                progress_label_continue.set_text(f"❌ Error: {str(e)[:50]}...")
                progress_bar_continue.value = 0
                log_output_continue.push(f"\n❌ An error occurred: {e}")
                ui.notify(f"Error continuing project: {e}", color="negative", multi_line=True)
            finally:
                continue_button.enable()
                continue_button.set_text("▶️ Continue Project")

        with util_content:
            topic_folders_util = get_topic_folders("output")
            if not topic_folders_util:
                with ui.card().classes("w-full").style("padding: 60px; text-align: center;"):
                    ui.icon("folder_open", size="xl").classes("text-gray-400")
                    ui.label("No Projects Found").classes("text-2xl font-semibold text-gray-700 dark:text-gray-300 mt-4")
                    ui.label("Create a project first to use utilities").classes("text-gray-500 dark:text-gray-400 mt-2")
                return
            
            # Get project statuses
            all_statuses = [video_generator.check_theorem_status({"theorem": th}) for th in topic_folders_util]
            incomplete_projects = [s for s in all_statuses if not s["has_combined_video"]]
            complete_projects = [s for s in all_statuses if s["has_combined_video"]]
            
            # Continue Unfinished Projects Section
            with ui.card().classes("w-full mb-4").style("padding: 24px;"):
                with ui.row().classes("w-full items-center gap-3 mb-4"):
                    ui.icon("play_circle", size="lg").classes("text-primary")
                    with ui.column().classes("gap-1"):
                        ui.label("Continue Unfinished Projects").classes("text-xl font-bold text-gray-900 dark:text-white")
                        ui.label("Resume generation for incomplete projects").classes("text-sm text-gray-600 dark:text-gray-400")
                
                if incomplete_projects:
                    continue_select = ui.select(
                        [p["topic"] for p in incomplete_projects],
                        label="Select Project to Continue"
                    ).props("outlined").classes("w-full")
                    
                    # Show project status
                    with ui.row().classes("w-full items-center gap-2 mt-2 mb-3"):
                        ui.icon("info", size="sm").classes("text-blue-500")
                        status_label = ui.label("Select a project to see its status").classes("text-sm text-gray-600 dark:text-gray-400")
                    
                    def update_status():
                        if continue_select.value:
                            status = next((s for s in incomplete_projects if s["topic"] == continue_select.value), None)
                            if status:
                                rendered = status["rendered_scenes"]
                                total = status["total_scenes"]
                                status_label.set_text(f"Progress: {rendered}/{total} scenes rendered ({rendered/total*100:.0f}%)")
                    
                    continue_select.on_value_change(lambda: update_status())
                    
                    continue_button = ui.button(
                        "▶️ Continue Project",
                        on_click=lambda: handle_continue(continue_select.value),
                    ).props("unelevated no-caps").classes("w-full mt-2")
                    
                    # Progress section for continue
                    progress_container_continue = ui.column().classes("w-full gap-3 mt-4").style("display: none;")
                    with progress_container_continue:
                        progress_label_continue = ui.label("Starting...").classes("text-sm font-medium text-gray-700 dark:text-gray-300")
                        progress_bar_continue = ui.linear_progress(value=0, show_value=False).props("color=primary size=8px").classes("w-full")
                        
                        with ui.expansion("View Detailed Logs", icon="terminal").props("duration=600").classes("w-full mt-2"):
                            log_output_continue = ui.log().props("id=log_output_continue").classes(
                                "w-full h-64 bg-gray-900 dark:bg-black text-white font-mono rounded-lg"
                            )
                else:
                    with ui.row().classes("w-full items-center gap-2 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg"):
                        ui.icon("check_circle", size="md").classes("text-green-600")
                        ui.label("All projects are complete! 🎉").classes("text-green-700 dark:text-green-300 font-medium")

    # --- Main Content Area (UI Definition) ---
    with ui.column().classes("w-full max-w-6xl mx-auto gap-6").style("padding: 24px 16px;"):
        with ui.tabs().classes("w-full mb-4") as main_tabs:
            one = ui.tab("✨ Generate").props("no-caps")
            two = ui.tab("📊 Dashboard").props("no-caps")
            three = ui.tab("📺 Project View").props("no-caps")
            four = ui.tab("🔧 Utilities").props("no-caps")

        with ui.tab_panels(main_tabs, value=one).classes("w-full"):
            with ui.tab_panel(one):
                # Modern header with gradient background
                with ui.card().classes("w-full shadow-lg border-0 mb-6").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); padding: 24px;"):
                    with ui.row().classes("w-full items-center gap-3"):
                        ui.icon("auto_awesome", size="xl").classes("text-white")
                        with ui.column().classes("gap-1"):
                            ui.label("Generate").classes("text-3xl font-bold text-white")
                            ui.label("Create educational videos with AI-powered animations").classes("text-white/80")
                
                # Main Form
                with ui.card().classes("w-full").style("padding: 32px;"):
                    with ui.column().classes("w-full gap-3"):
                        # Project Name and Tips Row
                        with ui.row().classes("w-full gap-4 items-end"):
                            # Project Name Section (left side)
                            with ui.column().classes("flex-grow gap-2"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.icon("label", size="sm").classes("text-primary")
                                    ui.label("Project Name").classes("text-sm font-semibold text-gray-900 dark:text-white")
                                topic_input = (
                                    ui.input(placeholder="e.g., Binary Search, Bubble Sort, Linked Lists...")
                                    .props("outlined dense")
                                    .classes("w-full")
                                )
                                topic_input.value = "Binary Search"
                            
                            # Tips Card (right side) - aligned to bottom
                            with ui.card().classes("flex-shrink-0").style("padding: 10px 14px; background-color: #fffbeb; border: 1px solid #fef3c7; width: 320px;"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.icon("tips_and_updates", size="sm").classes("text-primary")
                                    ui.label("More details = better video!").classes("text-xs font-medium")
                        
                        # Description Section
                        with ui.column().classes("w-full").style("gap: 8px;"):
                            with ui.row().classes("items-center gap-2 justify-between w-full"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.icon("description", size="sm").classes("text-primary")
                                    ui.label("Video Description").classes("text-sm font-semibold text-gray-900 dark:text-white")
                                auto_gen_button = (
                                    ui.button("Auto-Generate", icon="auto_awesome")
                                    .props("flat dense no-caps")
                                    .classes("text-primary")
                                    .style("font-size: 0.875rem;")
                                    .tooltip("Generate description from project name using AI")
                                )
                            desc_input = (
                                ui.textarea(
                                    placeholder="""Tips for better videos:
• Be specific about the algorithm and target audience level (beginner/intermediate/advanced)
• Mention if you want code examples, analogies, or visualizations
• Include the key concepts you want to emphasize
• Specify the programming language if showing code
• Mention any real-world applications or use cases"""
                                )
                                .props("outlined rows=15")
                                .classes("w-full")
                                .style("margin-bottom: 12px;")
                            )
                            desc_input.value = """Create a short video for beginners explaining Binary Search algorithm.

Target Audience: Complete beginners to algorithms
Key Concepts: Divide and conquer, sorted arrays, logarithmic time complexity

Content:
• Start with a sorted array example: [1, 3, 5, 7, 9, 11, 13]
• Show step-by-step how to find a target number (e.g., 7)
• Visualize checking the middle element
• Demonstrate eliminating half the array each time
• Compare with linear search to show efficiency
• End with time complexity: O(log n) vs O(n)

Style: Use simple animations, friendly tone, and a relatable analogy (like guessing a number between 1-100)"""
                            
                            # Action Section - tight to textarea
                            with ui.row().classes("w-full justify-end"):
                                generate_button = (
                                    ui.button("Generate Video", icon="play_circle")
                                    .props("unelevated no-caps")
                                    .classes("px-6")
                                    .style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); color: white;")
                                )
                
                # Progress Indicator (hidden by default)
                progress_container = ui.column().classes("w-full gap-3 mt-6").style("display: none;")
                with progress_container:
                    with ui.card().classes("w-full shadow-lg border-l-4 border-primary").style("padding: 24px;"):
                        with ui.column().classes("w-full gap-4"):
                            with ui.row().classes("w-full items-center gap-3"):
                                ui.spinner(size="lg", color="primary")
                                progress_label = ui.label("Generating video...").classes("text-lg font-semibold text-gray-900 dark:text-white")
                            
                            progress_bar = ui.linear_progress(value=0, show_value=False).props('rounded color="primary"').style("height: 8px;")
                            
                            ui.label("This may take several minutes depending on the complexity of your project.").classes("text-sm text-gray-600 dark:text-gray-400")
                            
                            # Expandable log section
                            with ui.expansion("View Detailed Logs", icon="terminal").props("duration=600").classes("w-full mt-2"):
                                log_output = (
                                    ui.log()
                                    .props("id=log_output")
                                    .classes(
                                        "w-full h-64 bg-gray-900 dark:bg-black text-white font-mono rounded-lg"
                                    )
                                )
                
                # Define the auto-generate description handler
                async def handle_auto_generate_description():
                    topic = topic_input.value
                    if not topic or not topic.strip():
                        ui.notify("Please enter a Project Name first.", color="warning")
                        return
                    
                    # Disable button and show loading state
                    auto_gen_button.disable()
                    auto_gen_button.props("loading")
                    ui.notify("Generating description...", color="info")
                    
                    try:
                        # Run LLM call in executor to keep UI responsive
                        import concurrent.futures
                        
                        def generate_description():
                            # Use the planner model to generate description
                            llm = LiteLLMWrapper(
                                model_name=app_state["planner_model_name"],
                                temperature=0.7,
                                print_cost=False,
                                verbose=False,
                                use_langfuse=False
                            )
                            
                            # Create prompt that produces clean output without asterisks or preambles
                            prompt = f"""Generate a detailed video description for an educational video about "{topic}".

IMPORTANT INSTRUCTIONS:
- Start directly with the content, NO preambles like "Here is..." or "Of course!"
- Use plain text formatting with line breaks, NOT markdown asterisks or bold
- Use bullet points with • symbol, not asterisks
- Be specific, include concrete examples
- Make it educational and engaging

REQUIRED FORMAT:

Create a short video for [audience level] explaining [Topic].

Target Audience: [Specify audience level and background]
Key Concepts: [List 2-4 main concepts]

Content:
• [First key point with specific example]
• [Second key point with visualization approach]
• [Third key point showing step-by-step process]
• [Fourth key point with comparison or application]
• [Final point with complexity or summary]

Style: [Describe the teaching approach and tone]

Now generate the description for "{topic}" following this exact format."""

                            messages = [{"type": "text", "content": prompt}]
                            return llm(messages, metadata={})
                        
                        # Run in thread pool to avoid blocking UI
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            generated_desc = await loop.run_in_executor(executor, generate_description)
                        
                        if generated_desc and not generated_desc.startswith("Error"):
                            # Clean up any remaining asterisks or unwanted formatting
                            generated_desc = generated_desc.strip()
                            # Remove common preambles
                            preambles = [
                                "Of course! Here is",
                                "Here is a detailed",
                                "Here's a detailed",
                                "Sure! Here is",
                                "Certainly! Here is",
                            ]
                            for preamble in preambles:
                                if generated_desc.startswith(preamble):
                                    # Find the first newline after preamble and start from there
                                    first_newline = generated_desc.find('\n')
                                    if first_newline > 0:
                                        generated_desc = generated_desc[first_newline:].strip()
                                    break
                            
                            desc_input.value = generated_desc
                            ui.notify("Description generated successfully!", color="positive")
                        else:
                            ui.notify("Failed to generate description. Please try again.", color="negative")
                    
                    except Exception as e:
                        print(f"Error generating description: {e}")
                        ui.notify(f"Error: {str(e)}", color="negative")
                    
                    finally:
                        # Re-enable button
                        auto_gen_button.enable()
                        auto_gen_button.props(remove="loading")
                
                # Connect the auto-generate button handler
                auto_gen_button.on_click(handle_auto_generate_description)
                
                # Define the generate handler now that UI elements exist
                async def handle_generate():
                    topic = topic_input.value
                    description = desc_input.value
                    if not topic or not description:
                        ui.notify(
                            "Please provide both a Project Name and Description.", color="warning"
                        )
                        return
                    
                    # Check if project already exists
                    project_path = get_project_path("output", topic)
                    is_existing = os.path.exists(project_path)
                    
                    # Disable button and show progress
                    generate_button.disable()
                    progress_container.style("display: block;")
                    progress_bar.set_visibility(True)
                    progress_label.set_text("🚀 Starting generation...")
                    progress_bar.value = 0
                    log_output.clear()
                    
                    if is_existing:
                        log_output.push(f"🔄 Resuming existing project: '{topic}'")
                        log_output.push(f"✅ Checking project status and continuing from where it left off...")
                    else:
                        log_output.push(f"🚀 Starting new project: '{topic}'")
                        log_output.push(f"📋 Initializing video generation pipeline...")
                    
                    try:
                        # Don't modify output_dir - generate_video.py handles folder structure
                        
                        # Update progress: Planning phase
                        progress_label.set_text("📝 Planning video structure...")
                        progress_bar.value = 0.1
                        await asyncio.sleep(0.1)  # Allow UI to update
                        
                        # Start generation in background thread to keep UI responsive
                        import concurrent.futures
                        import threading
                        import sys
                        from io import StringIO
                        
                        # Shared state for progress updates
                        generation_complete = threading.Event()
                        generation_error = [None]
                        current_progress = {"value": 0.1, "text": "Starting..."}
                        log_buffer = []  # Buffer for log messages
                        progress_lock = threading.Lock()
                        
                        def progress_callback(value, text):
                            """Thread-safe progress callback"""
                            with progress_lock:
                                current_progress["value"] = value
                                current_progress["text"] = text
                        
                        class LogCapture:
                            """Capture stdout/stderr and store in buffer"""
                            def __init__(self, original_stream, buffer_list):
                                self.original_stream = original_stream
                                self.buffer_list = buffer_list
                                self.skip_patterns = [
                                    "Langfuse client is disabled",
                                    "LANGFUSE_PUBLIC_KEY",
                                    "No video folders found",
                                    "langfuse.com/docs",
                                    "See our docs:",
                                    "==> Loading existing implementation plans",
                                    "<== Finished loading plans",
                                    "Found: 0, Missing:",
                                    "Generating missing implementation plans for scenes:",
                                    "Loaded existing topic session ID:",
                                    "Saved topic session ID to",
                                ]
                                
                            def write(self, text):
                                self.original_stream.write(text)  # Still print to console
                                if text.strip():  # Only add non-empty lines
                                    # Filter out unnecessary technical messages
                                    should_skip = any(pattern in text for pattern in self.skip_patterns)
                                    if not should_skip:
                                        # Transform technical messages to user-friendly ones
                                        cleaned_text = text.rstrip()
                                        
                                        # Replace technical terms
                                        replacements = {
                                            "STARTING VIDEO PIPELINE FOR TOPIC:": "🚀 Starting video generation for:",
                                            "[PHASE 1: SCENE OUTLINE]": "📝 Phase 1: Planning video structure",
                                            "[PHASE 1 COMPLETE]": "✅ Phase 1: Video structure ready",
                                            "[PHASE 2: IMPLEMENTATION PLANS]": "🎨 Phase 2: Designing all scenes",
                                            "[PHASE 2 COMPLETE]": "✅ Phase 2: All scenes designed",
                                            "[PHASE 3: CODE GENERATION & RENDERING (SCENE-BY-SCENE)]": "🎬 Phase 3: Rendering all scenes",
                                            "[PHASE 3 COMPLETE]": "✅ Phase 3: All scenes rendered",
                                            "PIPELINE FINISHED FOR TOPIC:": "✅ Video generation complete for:",
                                            "scene outline saved": "✅ Video structure saved",
                                            "Loaded existing scene outline": "✅ Using existing video structure",
                                            "Loaded existing topic session ID": "",
                                            "Total Cost:": "💰 Cost:",
                                            "==> Generating scene outline": "📝 Starting: Generating video structure",
                                            "==> Scene outline generated": "✅ Finished: Video structure generated",
                                            "==> Generating scene implementations": "🎨 Starting: Planning scene details",
                                            "==> All concurrent scene implementations generated": "✅ Finished: All scene details planned",
                                            "==> Preparing to render": "💻 Starting: Scene rendering",
                                            "Starting concurrent processing": "💻 Processing scenes concurrently",
                                            "<== All scene processing tasks completed": "✅ Finished: All scenes processed",
                                        }
                                        
                                        # Apply replacements
                                        for old, new in replacements.items():
                                            if old in cleaned_text:
                                                cleaned_text = cleaned_text.replace(old, new)
                                        
                                        # Handle "Found X scenes in outline" dynamically
                                        import re
                                        scenes_match = re.search(r'Found (\d+) scenes? in outline', cleaned_text)
                                        if scenes_match:
                                            num_scenes = scenes_match.group(1)
                                            scene_word = "scene" if num_scenes == "1" else "scenes"
                                            cleaned_text = f"📊 Video has {num_scenes} {scene_word}"
                                        
                                        # Clean up scene prefixes like "[Binary Search | Scene 2]"
                                        scene_prefix_match = re.search(r'\[([^\]]+) \| Scene (\d+)\]', cleaned_text)
                                        if scene_prefix_match:
                                            scene_num = scene_prefix_match.group(2)
                                            cleaned_text = re.sub(r'\[([^\]]+) \| Scene (\d+)\]', f'Scene {scene_num}:', cleaned_text)
                                        
                                        # Clean up topic prefixes like "[Binary Search]"
                                        cleaned_text = re.sub(r'\[([^\]]+)\]', '', cleaned_text).strip()
                                        
                                        # Remove excessive equals signs and empty messages
                                        if "======================================================" in cleaned_text or not cleaned_text.strip():
                                            return
                                        
                                        with progress_lock:
                                            self.buffer_list.append(cleaned_text)
                                        
                            def flush(self):
                                self.original_stream.flush()
                        
                        def run_generation_sync():
                            """Run generation in a separate thread"""
                            # Capture stdout/stderr
                            old_stdout = sys.stdout
                            old_stderr = sys.stderr
                            
                            try:
                                # Redirect output to our capture
                                sys.stdout = LogCapture(old_stdout, log_buffer)
                                sys.stderr = LogCapture(old_stderr, log_buffer)
                                
                                # Create a new event loop for this thread
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # Run the generation pipeline with progress callback
                                loop.run_until_complete(
                                    video_generator.generate_video_pipeline(
                                        topic, 
                                        description, 
                                        max_retries=app_state["max_retries"],
                                        progress_callback=progress_callback
                                    )
                                )
                                loop.close()
                            except Exception as e:
                                generation_error[0] = e
                            finally:
                                # Restore original stdout/stderr
                                sys.stdout = old_stdout
                                sys.stderr = old_stderr
                                generation_complete.set()
                        
                        # Start generation in background thread
                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        generation_future = executor.submit(run_generation_sync)
                        
                        heartbeat_counter = 0
                        last_log_count = 0
                        last_progress_text = ""
                        
                        # Keep UI responsive and update progress from callback
                        while not generation_complete.is_set():
                            await asyncio.sleep(0.5)  # Check twice per second for smoother updates
                            heartbeat_counter += 1
                            
                            # Update UI with actual progress from pipeline
                            with progress_lock:
                                progress_bar.value = current_progress["value"]
                                
                                # Update progress label with elapsed time
                                elapsed = heartbeat_counter // 2
                                if current_progress["text"]:
                                    progress_label.set_text(f"{current_progress['text']} ({elapsed}s)")
                                else:
                                    progress_label.set_text(f"⏳ Working... ({elapsed}s)")
                                
                                # Push new log messages to UI
                                if len(log_buffer) > last_log_count:
                                    for log_line in log_buffer[last_log_count:]:
                                        log_output.push(log_line)
                                    last_log_count = len(log_buffer)
                                    last_progress_text = current_progress["text"]
                        
                        # Check for errors
                        if generation_error[0]:
                            raise generation_error[0]
                        
                        # Wait for thread to fully complete
                        generation_future.result()
                        
                        # Push any remaining logs
                        with progress_lock:
                            if len(log_buffer) > last_log_count:
                                for log_line in log_buffer[last_log_count:]:
                                    log_output.push(log_line)
                                last_log_count = len(log_buffer)
                        
                        # Finalization phase
                        progress_label.set_text("🎞️ Combining videos and creating subtitles...")
                        progress_bar.value = 0.9
                        
                        # Check if any scenes failed
                        has_failures = len(video_generator.failed_scenes) > 0
                        
                        if has_failures:
                            log_output.push(f"\n⚠️ Pipeline completed with {len(video_generator.failed_scenes)} failed scene(s). Finalizing project...")
                        else:
                            log_output.push("\n✅ Pipeline completed! Finalizing project...")
                        
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(None, combine_videos, topic)
                        
                        # Complete
                        progress_bar.value = 1.0
                        
                        if has_failures:
                            progress_label.set_text(f"⚠️ Video generation completed with {len(video_generator.failed_scenes)} failed scene(s)")
                            log_output.push(f"⚠️ Project finalized with {len(video_generator.failed_scenes)} scene(s) failed. Check logs for details.")
                            ui.notify(
                                f"⚠️ Video for '{topic}' completed with failures. Check logs.",
                                color="warning",
                                icon="warning",
                            )
                        else:
                            progress_label.set_text("✅ Video generation complete!")
                            log_output.push("✅ Project finalized successfully!")
                            if result == "success":
                                ui.notify(
                                    f"Video for '{topic}' generated!",
                                    color="positive",
                                    icon="check_circle",
                                )
                            elif result == "already_exists":
                                ui.notify(f"ℹ️ Video for '{topic}' already finalized", color="info", icon="info")
                            elif result == "no_scenes":
                                ui.notify(f"⚠️ No rendered scenes found for '{topic}'", color="warning", icon="warning")
                        
                        # Hide progress after 2 seconds
                        await asyncio.sleep(2)
                        progress_container.style("display: none;")
                        
                    except Exception as e:
                        progress_label.set_text(f"❌ Error: {str(e)[:50]}...")
                        progress_bar.value = 0
                        log_output.push(f"\n❌ An error occurred: {e}")
                        ui.notify(f"An error occurred: {e}", color="negative", multi_line=True)
                    finally:
                        generate_button.enable()
                        await update_dashboard()
                
                # Connect the button to the handler
                generate_button.on_click(handle_generate)
            with ui.tab_panel(two):
                with ui.column().classes("w-full gap-6"):
                    # Modern header with gradient background
                    with ui.card().classes("w-full shadow-lg border-0").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); padding: 24px;"):
                        with ui.row().classes("w-full items-center justify-between"):
                            with ui.row().classes("items-center gap-3"):
                                ui.icon("dashboard", size="xl").classes("text-white")
                                with ui.column().classes("gap-1"):
                                    ui.label("Dashboard").classes("text-3xl font-bold text-white")
                                    ui.label("Monitor your projects and track progress").classes("text-white/80")
                            ui.button("Refresh", on_click=update_dashboard, icon="refresh").props(
                                "flat round"
                            ).classes("text-white").tooltip("Refresh dashboard")
                    
                    dashboard_content = ui.column().classes("w-full gap-4 mt-4")
            with ui.tab_panel(three):
                with ui.column().classes("w-full gap-6"):
                    # Modern header with gradient background
                    with ui.card().classes("w-full shadow-lg border-0").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); padding: 24px;"):
                        with ui.row().classes("w-full items-center gap-3"):
                            ui.icon("tv", size="xl").classes("text-white")
                            with ui.column().classes("gap-1"):
                                ui.label("Project View").classes("text-3xl font-bold text-white")
                                ui.label("Watch videos, explore code, and learn with AI").classes("text-white/80")
                    
                    topic_folders = get_topic_folders("output")
                    default_topic = app_state.get("selected_topic") or (
                        topic_folders[0] if topic_folders else None
                    )

                    # Project selector with modern styling
                    with ui.card().classes("w-full shadow-md"):
                        with ui.row().classes("w-full items-center gap-3 p-4"):
                            ui.icon("folder_open", size="lg").classes("text-primary")
                            inspector_select = (
                                ui.select(
                                    topic_folders,
                                    label="Select a Project",
                                    value=default_topic,
                                    on_change=lambda e: update_inspector(e.value),
                                )
                                .props("outlined")
                                .classes("flex-grow text-lg")
                                .style("font-size: 1.125rem; min-height: 56px;")
                            )
                    
                    inspector_content = ui.column().classes("w-full gap-6 mt-4")
            
            with ui.tab_panel(four):
                with ui.column().classes("w-full gap-6"):
                    # Modern header with gradient background
                    with ui.card().classes("w-full shadow-lg border-0").style("background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%); padding: 24px;"):
                        with ui.row().classes("w-full items-center gap-3"):
                            ui.icon("build_circle", size="xl").classes("text-white")
                            with ui.column().classes("gap-1"):
                                ui.label("Utilities").classes("text-3xl font-bold text-white")
                                ui.label("Continue and manage your video projects").classes("text-white/80")
                    
                    util_content = ui.column().classes("w-full mt-4")
    
    # --- Define inspect_project after all UI elements are created ---
    async def inspect_project(topic_name):
        app_state["selected_topic"] = topic_name
        inspector_select.set_value(topic_name)
        await update_inspector(topic_name)
        main_tabs.set_value("📺 Project View")

    # --- Initial UI State Population ---
    await update_dashboard()
    
    # Initialize inspector with default topic if available
    topic_folders = get_topic_folders("output")
    default_topic = app_state.get("selected_topic") or (
        topic_folders[0] if topic_folders else None
    )
    if default_topic:
        app_state["selected_topic"] = default_topic
        await update_inspector(default_topic)
    
    update_util_tab()


# Run the app with increased WebSocket timeout to prevent disconnections during long operations
ui.run(
    title='AlgoVision',
    reconnect_timeout=3600.0,  # 5 minutes timeout instead of default 30 seconds
    reload=False,  # Disable auto-reload to prevent interruption of Manim rendering
)

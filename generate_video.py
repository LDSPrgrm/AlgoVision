# -*- coding: utf-8 -*-

# Configure UTF-8 encoding for stdout/stderr on Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import random
from typing import Union, List, Dict, Optional
import subprocess
import argparse
import glob
from PIL import Image
import re
from dotenv import load_dotenv
import asyncio
import uuid  # Import uuid for generating trace_id

from utils.litellm import LiteLLMWrapper
from utils.utils import (
    _prepare_text_inputs,
)  # Keep _prepare_text_inputs if still used directly in main

# Import new modules
from src.core.video_planner import VideoPlanner
from src.core.code_generator import CodeGenerator
from src.core.video_renderer import VideoRenderer
from src.utils.utils import (
    _print_response,
    _extract_code,
    extract_xml,
)  # Import utility functions
from src.config.config import Config  # Import Config class
from src.utils.error_recovery import ErrorRecovery  # Import error recovery

# Video parsing
from src.core.parse_video import get_images_from_video, image_with_most_non_black_space
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


class VideoGenerator:
    """
    A class for generating manim videos using AI models.

    This class coordinates the video generation pipeline by managing scene planning,
    code generation, and video rendering. It supports concurrent scene processing,
    visual code fixing, and RAG (Retrieval Augmented Generation).

    Args:
        planner_model: Model used for scene planning and high-level decisions
        scene_model: Model used specifically for scene generation (defaults to planner_model)
        output_dir (str): Directory to store generated files and videos
        verbose (bool): Whether to print detailed output
        use_rag (bool): Whether to use Retrieval Augmented Generation
        use_context_learning (bool): Whether to use context learning with example code
        context_learning_path (str): Path to context learning examples
        chroma_db_path (str): Path to ChromaDB for RAG
        manim_docs_path (str): Path to Manim documentation for RAG
        embedding_model (str): Model to use for embeddings
        use_visual_fix_code (bool): Whether to use visual feedback for code fixing
        use_langfuse (bool): Whether to enable Langfuse logging
        trace_id (str, optional): Trace ID for logging
        max_scene_concurrency (int): Maximum number of scenes to process concurrently

    Attributes:
        output_dir (str): Directory for output files
        verbose (bool): Verbosity flag
        use_visual_fix_code (bool): Visual code fixing flag
        session_id (str): Unique session identifier
        scene_semaphore (asyncio.Semaphore): Controls concurrent scene processing
        banned_reasonings (list): List of banned reasoning patterns
        planner (VideoPlanner): Handles scene planning
        code_generator (CodeGenerator): Handles code generation
        video_renderer (VideoRenderer): Handles video rendering
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
        self.failed_scenes = []  # Track failed scenes for reporting
        self.error_recovery = ErrorRecovery(output_dir)  # Initialize error recovery
        self.rate_limit_detected = False  # Track if we hit rate limits
        self.last_rate_limit_time = 0  # Track when we last hit rate limit

        # Initialize separate modules
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
        """
        Load existing session ID from file or create a new one.

        Returns:
            str: The session ID either loaded from file or newly created.
        """
        session_file = os.path.join(self.output_dir, "session_id.txt")
        # print(f"Checking for session ID file at: {session_file}")

        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_id = f.read().strip()
                # print(f"Loaded existing session ID: {session_id}")
                return session_id

        # Create new session ID if none exists
        session_id = str(uuid.uuid4())
        print(f"No existing session ID found. Creating a new one: {session_id}")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(session_file, "w", encoding='utf-8') as f:
            f.write(session_id)
        print(f"Saved new session ID to {session_file}")
        return session_id

    def _save_topic_session_id(self, topic: str, session_id: str) -> None:
        """
        Save session ID for a specific topic.

        Args:
            topic (str): The topic to save the session ID for
            session_id (str): The session ID to save
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        # Create structure: output/Topic Name/topic_name/
        topic_dir = os.path.join(self.output_dir, topic, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)

        session_file = os.path.join(topic_dir, "session_id.txt")
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(session_id)

    def _load_topic_session_id(self, topic: str) -> Optional[str]:
        """
        Load session ID for a specific topic if it exists.

        Args:
            topic (str): The topic to load the session ID for

        Returns:
            Optional[str]: The session ID if found, None otherwise
        """
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)
        # Look in structure: output/Topic Name/topic_name/
        session_file = os.path.join(self.output_dir, topic, file_prefix, "session_id.txt")

        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                session_id = f.read().strip()
                print(
                    f"Loaded session ID '{session_id}' for topic '{topic}' from: {session_file}"
                )
                return session_id
        print(f"No specific session ID found for topic '{topic}'.")
        return None

    def validate_and_clean_existing_content(self, topic: str) -> dict:
        """
        Validate all existing content for a topic before resuming.
        Cleans corrupted files and reports what was found.
        
        Args:
            topic (str): The topic to validate
            
        Returns:
            dict: Summary of validation results
        """
        print(f"\n{'='*70}")
        print(f"VALIDATING EXISTING CONTENT FOR: {topic}")
        print(f"{'='*70}")
        
        all_errors = self.error_recovery.scan_topic_for_errors(topic)
        
        if not all_errors:
            print("✓ All existing content is valid. Safe to resume.")
            print(f"{'='*70}\n")
            return {"valid": True, "cleaned_files": 0, "errors": {}}
        
        # Found errors - report and clean
        print(f"⚠️  Found corrupted files in {len(all_errors)} location(s):")
        
        total_cleaned = 0
        for scene_num, errors in all_errors.items():
            scene_label = "Scene Outline" if scene_num == 0 else f"Scene {scene_num}"
            print(f"\n  {scene_label}:")
            for error_type, files in errors.items():
                print(f"    - {error_type}: {len(files)} file(s)")
                total_cleaned += len(files)
        
        # Check if errors are rate limit related
        rate_limit_errors = any(
            'rate' in error_type.lower() or '429' in error_type
            for errors in all_errors.values()
            for error_type in errors.keys()
        )
        
        if rate_limit_errors:
            print(f"\n⚠️  RATE LIMIT DETECTED in existing files!")
            print(f"   Recommendation: Wait before regenerating to avoid hitting limits again.")
            self.rate_limit_detected = True
            import time
            self.last_rate_limit_time = time.time()
        
        print(f"\nCleaning {total_cleaned} corrupted file(s)...")
        deleted_files = self.error_recovery.clean_corrupted_files(topic, dry_run=False)
        
        print(f"✓ Cleaned {len(deleted_files)} file(s). Missing content will be regenerated.")
        print(f"{'='*70}\n")
        
        return {
            "valid": False,
            "cleaned_files": len(deleted_files),
            "errors": all_errors,
            "rate_limit_detected": rate_limit_errors
        }
    
    def check_rate_limit_backoff(self, min_wait_seconds: int = 60):
        """
        Check if we should wait before making API calls due to rate limits.
        
        Args:
            min_wait_seconds (int): Minimum seconds to wait after rate limit
            
        Returns:
            bool: True if we should proceed, False if we should wait
        """
        if not self.rate_limit_detected:
            return True
        
        import time
        elapsed = time.time() - self.last_rate_limit_time
        
        if elapsed < min_wait_seconds:
            wait_time = int(min_wait_seconds - elapsed)
            print(f"\n⏳ Rate limit detected. Waiting {wait_time} seconds before continuing...")
            print(f"   (To avoid immediate re-failure)")
            time.sleep(wait_time)
            print(f"✓ Wait complete. Proceeding with regeneration.\n")
        
        # Reset flag after waiting
        self.rate_limit_detected = False
        return True
    
    def handle_generation_error(self, error_message: str, attempt: int = 1, max_attempts: int = 3):
        """
        Handle errors during content generation with exponential backoff.
        
        Args:
            error_message (str): The error message from the LLM
            attempt (int): Current attempt number
            max_attempts (int): Maximum number of retry attempts
            
        Returns:
            bool: True if should retry, False if should abort
        """
        import time
        
        error_lower = error_message.lower()
        
        # Check if it's a rate limit error
        is_rate_limit = any(pattern in error_lower for pattern in [
            'rate limit', '429', 'quota exceeded', 'resource exhausted'
        ])
        
        # Check if it's a temporary error
        is_temporary = any(pattern in error_lower for pattern in [
            'overloaded', '503', '502', 'timeout', 'deadline exceeded', '504'
        ])
        
        if is_rate_limit or is_temporary:
            if attempt >= max_attempts:
                print(f"\n❌ Max retry attempts ({max_attempts}) reached. Aborting.")
                return False
            
            # Exponential backoff: 60s, 120s, 240s
            wait_time = 60 * (2 ** (attempt - 1))
            
            error_type = "Rate limit" if is_rate_limit else "Temporary error"
            print(f"\n⏳ {error_type} detected (attempt {attempt}/{max_attempts})")
            print(f"   Error: {error_message[:100]}...")
            print(f"   Waiting {wait_time} seconds before retry...")
            
            time.sleep(wait_time)
            print(f"✓ Wait complete. Retrying...\n")
            
            self.rate_limit_detected = True
            self.last_rate_limit_time = time.time()
            return True
        
        # Not a retryable error
        return False

    def generate_scene_outline(
        self, topic: str, description: str, session_id: str
    ) -> str:
        """
        Generate scene outline using VideoPlanner.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            session_id (str): Session identifier for tracking

        Returns:
            str: Generated scene outline
        """
        print(f"[{topic}] ==> Generating scene outline...")
        
        # Check for corrupted outline from previous attempts
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        outline_path = os.path.join(self.output_dir, topic, file_prefix, f"{file_prefix}_scene_outline.txt")
        if os.path.exists(outline_path):
            is_corrupted, error_type = self.error_recovery.is_file_corrupted(outline_path)
            if is_corrupted:
                print(f"[{topic}] ⚠️  Detected corrupted scene outline ({error_type}). Deleting for regeneration...")
                
                # Check if it's a rate limit error
                if 'rate' in error_type.lower() or '429' in error_type:
                    self.rate_limit_detected = True
                    import time
                    self.last_rate_limit_time = time.time()
                
                os.remove(outline_path)
        
        # Check if we need to wait due to rate limits
        self.check_rate_limit_backoff()
        
        outline = self.planner.generate_scene_outline(topic, description, session_id)
        print(f"[{topic}] ==> Scene outline generated successfully.")
        return outline

    async def generate_scene_implementation(
        self, topic: str, description: str, plan: str, session_id: str
    ) -> List[str]:
        """
        Generate scene implementations using VideoPlanner.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            plan (str): The scene plan to implement
            session_id (str): Session identifier for tracking

        Returns:
            List[str]: List of generated scene implementations
        """
        print(f"[{topic}] ==> Generating scene implementations sequentially...")
        implementations = await self.planner.generate_scene_implementation(
            topic, description, plan, session_id
        )
        print(f"[{topic}] ==> Scene implementations generated.")
        return implementations

    async def generate_scene_implementation_concurrently(
        self, topic: str, description: str, plan: str, session_id: str
    ) -> List[str]:
        """
        Generate scene implementations concurrently using VideoPlanner.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            plan (str): The scene plan to implement
            session_id (str): Session identifier for tracking

        Returns:
            List[str]: List of generated scene implementations
        """
        print(f"[{topic}] ==> Generating scene implementations concurrently...")
        implementations = await self.planner.generate_scene_implementation_concurrently(
            topic, description, plan, session_id, self.scene_semaphore
        )  # Pass semaphore
        print(f"[{topic}] ==> All concurrent scene implementations generated.")
        return implementations

    def cleanup_invalid_success_markers(self, topic: str) -> int:
        """
        Remove succ_rendered.txt files for scenes that don't actually have rendered videos.
        
        Args:
            topic (str): The topic to check
            
        Returns:
            int: Number of invalid markers removed
        """
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
                # Check if video actually exists
                media_dir = os.path.join(topic_dir, "media", "videos")
                video_pattern = os.path.join(media_dir, f"{file_prefix}_scene{scene_num}_v*")
                video_folders = glob.glob(video_pattern)
                
                has_video = False
                for video_folder in video_folders:
                    # Check for actual .mp4 files
                    for res_dir in ["1080p60", "720p30", "480p15"]:
                        video_file = os.path.join(video_folder, res_dir, f"Scene{scene_num}.mp4")
                        if os.path.exists(video_file):
                            has_video = True
                            break
                    if has_video:
                        break
                
                if not has_video:
                    print(f"[{topic}] Removing invalid succ_rendered.txt for scene {scene_num} (no video found)")
                    os.remove(succ_file)
                    removed_count += 1
        
        return removed_count
    
    def validate_all_scene_plans(self, topic: str, scene_outline: str, skip_corruption_check: bool = False) -> Dict[str, any]:
        """
        Validate that ALL scene plans exist and are not corrupted before starting code generation.
        
        Args:
            topic (str): The topic to validate
            scene_outline (str): The scene outline content
            
        Returns:
            Dict with validation results and list of missing/corrupted scenes
        """
        print(f"\n{'='*70}")
        print(f"PRE-FLIGHT CHECK: Validating all scene plans for {topic}")
        print(f"{'='*70}")
        
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        scene_outline_content = extract_xml(scene_outline, "SCENE_OUTLINE")
        total_scenes = len(re.findall(r"<SCENE_(\d+)>[^<]", scene_outline_content))
        
        print(f"Total scenes expected: {total_scenes}")
        
        missing_plans = []
        corrupted_plans = []
        valid_plans = []
        
        # Check each scene's plans
        for scene_num in range(1, total_scenes + 1):
            scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{scene_num}")
            subplan_dir = os.path.join(scene_dir, "subplans")
            
            # Required plan files
            required_files = {
                'vision_storyboard': os.path.join(subplan_dir, f"{file_prefix}_scene{scene_num}_vision_storyboard_plan.txt"),
                'technical_implementation': os.path.join(subplan_dir, f"{file_prefix}_scene{scene_num}_technical_implementation_plan.txt"),
                'animation_narration': os.path.join(subplan_dir, f"{file_prefix}_scene{scene_num}_animation_narration_plan.txt"),
                'proto_tcm': os.path.join(scene_dir, "proto_tcm.json"),
                'combined_plan': os.path.join(scene_dir, f"{file_prefix}_scene{scene_num}_implementation_plan.txt")
            }
            
            scene_status = {
                'scene_number': scene_num,
                'missing': [],
                'corrupted': [],
                'valid': []
            }
            
            # Check each required file
            for file_type, file_path in required_files.items():
                if not os.path.exists(file_path):
                    scene_status['missing'].append(file_type)
                else:
                    if skip_corruption_check:
                        # Even when skipping full corruption check, still check for rate limit errors
                        # because those need immediate attention
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read(500)  # Just read first 500 chars
                                if 'rate' in content.lower() and ('limit' in content.lower() or '429' in content):
                                    scene_status['corrupted'].append((file_type, 'rate_limit_in_content'))
                                else:
                                    scene_status['valid'].append(file_type)
                        except:
                            scene_status['valid'].append(file_type)
                    else:
                        # Check if file is corrupted
                        is_corrupted, error_type = self.error_recovery.is_file_corrupted(file_path)
                        if is_corrupted:
                            scene_status['corrupted'].append((file_type, error_type))
                        else:
                            scene_status['valid'].append(file_type)
            
            # Categorize scene
            if scene_status['missing'] or scene_status['corrupted']:
                if scene_status['missing']:
                    missing_plans.append(scene_status)
                if scene_status['corrupted']:
                    corrupted_plans.append(scene_status)
            else:
                valid_plans.append(scene_num)
        
        # Print results
        print(f"\n✓ Valid scenes: {len(valid_plans)}/{total_scenes}")
        
        if missing_plans:
            print(f"\n⚠️  Scenes with missing plans: {len(missing_plans)}")
            for scene in missing_plans:
                print(f"  Scene {scene['scene_number']}: Missing {', '.join(scene['missing'])}")
        
        if corrupted_plans:
            print(f"\n⚠️  Scenes with corrupted plans: {len(corrupted_plans)}")
            for scene in corrupted_plans:
                print(f"  Scene {scene['scene_number']}:")
                for file_type, error_type in scene['corrupted']:
                    print(f"    - {file_type}: {error_type}")
        
        all_valid = len(valid_plans) == total_scenes
        
        if all_valid:
            print(f"\n✅ All scene plans are valid and ready for code generation!")
        else:
            print(f"\n❌ {total_scenes - len(valid_plans)} scene(s) need plan generation/regeneration")
        
        print(f"{'='*70}\n")
        
        return {
            'all_valid': all_valid,
            'total_scenes': total_scenes,
            'valid_count': len(valid_plans),
            'valid_scenes': valid_plans,
            'missing_plans': missing_plans,
            'corrupted_plans': corrupted_plans
        }
    
    async def ensure_all_scene_plans_ready(self, topic: str, description: str, scene_outline: str, session_id: str) -> bool:
        """
        Ensure ALL scene plans are generated and valid before starting code generation.
        Generates missing plans and cleans/regenerates corrupted ones.
        
        Args:
            topic (str): The topic
            description (str): Description
            scene_outline (str): Scene outline content
            session_id (str): Session ID
            
        Returns:
            bool: True if all plans are ready, False if failed
        """
        print(f"[{topic}] ==> Ensuring all scene plans are ready...")
        
        # Validate current state
        validation = self.validate_all_scene_plans(topic, scene_outline)
        
        if validation['all_valid']:
            print(f"[{topic}] ✓ All scene plans already exist and are valid.")
            return True
        
        # Clean corrupted plans
        if validation['corrupted_plans']:
            print(f"[{topic}] Cleaning corrupted plans...")
            for scene_status in validation['corrupted_plans']:
                scene_num = scene_status['scene_number']
                self.error_recovery.clean_corrupted_files(topic, scene_num, dry_run=False)
            
            # Check for rate limits
            has_rate_limit = any(
                'rate' in error_type.lower() or '429' in error_type
                for scene in validation['corrupted_plans']
                for _, error_type in scene['corrupted']
            )
            if has_rate_limit:
                self.check_rate_limit_backoff()
        
        # Get list of scenes that need plans
        scenes_needing_plans = set()
        for scene in validation['missing_plans']:
            scenes_needing_plans.add(scene['scene_number'])
        for scene in validation['corrupted_plans']:
            scenes_needing_plans.add(scene['scene_number'])
        
        if not scenes_needing_plans:
            return True
        
        print(f"[{topic}] Generating plans for {len(scenes_needing_plans)} scene(s): {sorted(scenes_needing_plans)}")
        
        # Generate missing/corrupted plans
        try:
            # Use the existing concurrent generation method (now we can await it)
            implementations = await self.planner.generate_scene_implementation_concurrently(
                topic, description, scene_outline, session_id, self.scene_semaphore
            )
            
            # Wait a moment for files to be fully written
            import time
            print(f"[{topic}] Waiting for files to be written...")
            time.sleep(3)  # Give filesystem time to flush
            
            # Validate again - skip corruption checks since files were just generated
            validation = self.validate_all_scene_plans(topic, scene_outline, skip_corruption_check=True)
            
            if validation['all_valid']:
                print(f"[{topic}] ✅ All scene plans successfully generated and validated!")
                return True
            else:
                print(f"[{topic}] ⚠️  Some plans still missing or corrupted after generation.")
                print(f"[{topic}] This may indicate persistent API issues.")
                return False
                
        except Exception as e:
            print(f"[{topic}] ❌ Error generating scene plans: {e}")
            return False
    
    def load_implementation_plans(self, topic: str) -> Dict[int, Optional[str]]:
        """
        Load implementation plans for each scene.

        Args:
            topic (str): The topic to load implementation plans for

        Returns:
            Dict[int, Optional[str]]: Dictionary mapping scene numbers to their plans.
                                    If a scene's plan is missing, its value will be None.
        """
        print(f"[{topic}] ==> Loading existing implementation plans...")
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)

        # Load scene outline from file
        # Use structure: output/Topic Name/topic_name/
        scene_outline_path = os.path.join(
            self.output_dir, topic, file_prefix, f"{file_prefix}_scene_outline.txt"
        )
        if not os.path.exists(scene_outline_path):
            print(
                f"[{topic}] WARNING: Scene outline not found. Cannot determine number of scenes to load."
            )
            return {}

        with open(scene_outline_path, "r") as f:
            scene_outline = f.read()

        # Extract scene outline to get number of scenes
        scene_outline_content = extract_xml(scene_outline, "SCENE_OUTLINE")
        scene_number = 0
        if scene_outline_content:
            scene_number = len(re.findall(r"<SCENE_(\d+)>[^<]", scene_outline_content))
        print(f"[{topic}] Found {scene_number} scenes in outline.")

        implementation_plans = {}
        found_count = 0
        missing_count = 0

        # Check each scene's implementation plan
        for i in range(1, scene_number + 1):
            plan_path = os.path.join(
                self.output_dir,
                topic,
                file_prefix,
                f"scene{i}",
                f"{file_prefix}_scene{i}_implementation_plan.txt",
            )
            if os.path.exists(plan_path):
                with open(plan_path, "r") as f:
                    implementation_plans[i] = f.read()
                if self.verbose:
                    print(f"[{topic}] Found existing implementation plan for scene {i}")
                found_count += 1
            else:
                implementation_plans[i] = None
                if self.verbose:
                    print(f"[{topic}] Missing implementation plan for scene {i}")
                missing_count += 1

        print(
            f"[{topic}] <== Finished loading plans. Found: {found_count}, Missing: {missing_count}."
        )
        return implementation_plans

    async def render_video_fix_code(
        self,
        topic: str,
        description: str,
        scene_outline: str,
        implementation_plans: List,
        max_retries=3,
        session_id: str = None,
    ) -> None:
        """
        Render the video for all scenes with code fixing capability.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            scene_outline (str): The overall scene outline
            implementation_plans (List): List of implementation plans for each scene
            max_retries (int, optional): Maximum number of code fix attempts. Defaults to 3.
            session_id (str, optional): Session identifier for tracking
        """
        print(
            f"[{topic}] ==> Preparing to render {len(implementation_plans)} scenes with code fixing enabled (max retries: {max_retries})."
        )
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)

        # Create tasks for each scene
        tasks = []
        for scene_num, implementation_plan in implementation_plans:
            i = scene_num - 1

            # Try to load scene trace id, or generate new one if it doesn't exist
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_num}")
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

            # --- NEW: Load the Proto-TCM for this scene ---
            proto_tcm_str = ""
            proto_tcm_path = os.path.join(
                self.output_dir, file_prefix, f"scene{scene_num}", "proto_tcm.json"
            )
            if os.path.exists(proto_tcm_path):
                with open(proto_tcm_path, "r") as f:
                    proto_tcm_str = f.read()
            else:
                print(
                    f"[{topic} | Scene {scene_num}] WARNING: proto_tcm.json not found. Code will be generated without timing constraints."
                )

            task = self.process_scene(
                i,
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

        # Execute all tasks concurrently
        print(f"[{topic}] Starting concurrent processing of {len(tasks)} scenes...")
        await asyncio.gather(*tasks)
        print(f"[{topic}] <== All scene processing tasks completed.")
        
        # Print summary report
        self._print_processing_summary(topic, len(implementation_plans))

    async def process_scene(
        self,
        i: int,
        scene_outline: str,
        scene_implementation: str,
        proto_tcm: str,
        topic: str,
        description: str,
        max_retries: int,
        file_prefix: str,
        session_id: str,
        scene_trace_id: str,
    ):  # added scene_trace_id
        """
        Process a single scene using CodeGenerator and VideoRenderer.

        Args:
            i (int): Scene index
            scene_outline (str): Overall scene outline
            scene_implementation (str): Implementation plan for this scene
            topic (str): The topic of the video
            description (str): Description of the video content
            max_retries (int): Maximum number of code fix attempts
            file_prefix (str): Prefix for file naming
            session_id (str): Session identifier for tracking
            scene_trace_id (str): Trace identifier for this scene
        """
        curr_scene = i + 1
        curr_version = 0
        rag_queries_cache = {}

        code_dir = os.path.join(
            self.output_dir, topic, file_prefix, f"scene{curr_scene}", "code"
        )
        os.makedirs(code_dir, exist_ok=True)
        media_dir = os.path.join(self.output_dir, topic, file_prefix, "media")

        async with self.scene_semaphore:
            print(f"[{topic} | Scene {curr_scene}] ---> Starting processing.")
            
            # Check for corrupted plan files before starting
            print(f"[{topic} | Scene {curr_scene}] Checking for corrupted plan files...")
            scene_errors = self.error_recovery.scan_scene_for_errors(topic, curr_scene)
            if scene_errors:
                print(f"[{topic} | Scene {curr_scene}] ⚠️  Detected corrupted files from previous LLM errors:")
                
                # Check if any are rate limit errors
                rate_limit_found = False
                for error_type, files in scene_errors.items():
                    print(f"  - {error_type}: {len(files)} file(s)")
                    if 'rate' in error_type.lower() or '429' in error_type:
                        rate_limit_found = True
                
                if rate_limit_found:
                    self.rate_limit_detected = True
                    import time
                    self.last_rate_limit_time = time.time()
                
                print(f"[{topic} | Scene {curr_scene}] Cleaning corrupted files for regeneration...")
                self.error_recovery.clean_corrupted_files(topic, curr_scene, dry_run=False)
                print(f"[{topic} | Scene {curr_scene}] ✓ Corrupted files cleaned. Scene will be regenerated.")
                
                # Wait if rate limit was detected
                if rate_limit_found:
                    self.check_rate_limit_backoff()

            # Step 3A: Generate initial manim code
            print(f"Scene {curr_scene} ---> Generating animation code...")
            code, log = self.code_generator.generate_manim_code(
                topic=topic,
                description=description,
                scene_outline=scene_outline,
                scene_implementation=scene_implementation,
                proto_tcm=proto_tcm,
                scene_number=curr_scene,
                additional_context=[
                    _prompt_manim_cheatsheet,
                    _code_font_size,
                    _code_limit,
                    _code_disable,
                ],
                scene_trace_id=scene_trace_id,
                session_id=session_id,
                rag_queries_cache=rag_queries_cache,
            )

            # Save initial code and log
            log_path = os.path.join(
                code_dir,
                f"{file_prefix}_scene{curr_scene}_v{curr_version}_init_log.txt",
            )
            code_path = os.path.join(
                code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py"
            )
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Scene {curr_scene} ✅ Animation code generated.")

            # Step 3B: Compile and fix code if needed
            print(f"Scene {curr_scene} ---> Rendering animation...")
            error_message = None
            retry_cycle = 0  # Track full retry cycles
            max_retry_cycles = 3  # Allow multiple full cycles
            
            while retry_cycle < max_retry_cycles:
                code, error_message = await self.video_renderer.render_scene(
                    code=code,
                    file_prefix=file_prefix,
                    curr_scene=curr_scene,
                    curr_version=curr_version,
                    code_dir=code_dir,
                    media_dir=media_dir,
                    max_retries=max_retries,
                    use_visual_fix_code=self.use_visual_fix_code,
                    visual_self_reflection_func=self.code_generator.visual_self_reflection,
                    banned_reasonings=self.banned_reasonings,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    session_id=session_id,
                )
                if error_message is None:
                    print(f"Scene {curr_scene} ✅ Animation rendered successfully.")
                    break
                
                if curr_version >= max_retries:
                    retry_cycle += 1
                    
                    if retry_cycle < max_retry_cycles:
                        # Clean up failed attempts and start fresh
                        print(f"Scene {curr_scene} ---> Regenerating code from scratch (attempt {retry_cycle + 1}/{max_retry_cycles})...")
                        
                        # Delete all failed code versions
                        for v in range(curr_version + 1):
                            failed_code = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{v}.py")
                            failed_log = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{v}_*.txt")
                            if os.path.exists(failed_code):
                                os.remove(failed_code)
                            # Remove associated logs
                            import glob as glob_module
                            for log_file in glob_module.glob(failed_log):
                                os.remove(log_file)
                        
                        # Reset version counter
                        curr_version = 0
                        
                        # Regenerate code from scratch with enhanced context
                        enhanced_context = [
                            _prompt_manim_cheatsheet,
                            _code_font_size,
                            _code_limit,
                            _code_disable,
                            f"\n\nIMPORTANT: Previous attempt failed with error: {error_message}\n"
                            f"Please avoid this error pattern and use simpler, more reliable Manim constructs.\n"
                            f"Focus on basic animations that are guaranteed to work."
                        ]
                        
                        code, log = self.code_generator.generate_manim_code(
                            topic=topic,
                            description=description,
                            scene_outline=scene_outline,
                            scene_implementation=scene_implementation,
                            proto_tcm=proto_tcm,
                            scene_number=curr_scene,
                            additional_context=enhanced_context,
                            scene_trace_id=scene_trace_id,
                            session_id=session_id,
                            rag_queries_cache=rag_queries_cache,
                        )
                        
                        # Save regenerated code
                        log_path = os.path.join(
                            code_dir,
                            f"{file_prefix}_scene{curr_scene}_v{curr_version}_regenerated_cycle{retry_cycle}_log.txt",
                        )
                        code_path = os.path.join(
                            code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py"
                        )
                        with open(log_path, "w", encoding="utf-8") as f:
                            f.write(log)
                        with open(code_path, "w", encoding="utf-8") as f:
                            f.write(code)
                        continue
                    else:
                        # All retry cycles exhausted
                        print(f"Scene {curr_scene} ⚠️ Failed to render after {max_retry_cycles} attempts. Check logs for details.")
                        # Save failure report
                        failure_report_path = os.path.join(
                            code_dir,
                            f"{file_prefix}_scene{curr_scene}_FAILED_REPORT.txt"
                        )
                        with open(failure_report_path, "w", encoding="utf-8") as f:
                            f.write(f"Scene {curr_scene} failed after {max_retry_cycles} retry cycles\n")
                            f.write(f"Total attempts: {curr_version + 1}\n")
                            f.write(f"Last error: {error_message}\n\n")
                            f.write(f"Suggestions:\n")
                            f.write(f"1. Manually review the implementation plan\n")
                            f.write(f"2. Simplify the scene requirements\n")
                            f.write(f"3. Check for Manim version compatibility issues\n")
                        
                        # Track failure for summary report
                        self.failed_scenes.append({
                            'topic': topic,
                            'scene': curr_scene,
                            'last_error': error_message,
                            'total_attempts': curr_version + 1,
                            'retry_cycles': max_retry_cycles
                        })
                        break

                curr_version += 1
                print(f"Scene {curr_scene} ---> Fixing code issues (attempt {curr_version + 1}/{max_retries})...")
                code, log = self.code_generator.fix_code_errors(
                    implementation_plan=scene_implementation,
                    proto_tcm=proto_tcm,
                    code=code,
                    error=error_message,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=curr_scene,
                    session_id=session_id,
                    rag_queries_cache=rag_queries_cache,
                )

                log_path = os.path.join(
                    code_dir,
                    f"{file_prefix}_scene{curr_scene}_v{curr_version}_fix_log.txt",
                )
                code_path = os.path.join(
                    code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py"
                )
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(log)
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(code)

    def _print_processing_summary(self, topic: str, total_scenes: int):
        """
        Print a summary report of scene processing results.
        
        Args:
            topic (str): The topic that was processed
            total_scenes (int): Total number of scenes attempted
        """
        successful_scenes = total_scenes - len(self.failed_scenes)
        
        print("\n" + "="*70)
        print(f"PROCESSING SUMMARY FOR: {topic}")
        print("="*70)
        print(f"Total Scenes: {total_scenes}")
        print(f"✓ Successful: {successful_scenes}")
        print(f"✗ Failed: {len(self.failed_scenes)}")
        
        if self.failed_scenes:
            print("\n" + "-"*70)
            print("FAILED SCENES DETAILS:")
            print("-"*70)
            for failure in self.failed_scenes:
                print(f"\n  Scene {failure['scene']}:")
                print(f"    Total Attempts: {failure['total_attempts']}")
                print(f"    Retry Cycles: {failure['retry_cycles']}")
                print(f"    Last Error: {failure['last_error'][:150]}...")
            print("\n" + "-"*70)
            print("RECOMMENDATIONS:")
            print("  1. Review failed scene implementation plans")
            print("  2. Check FAILED_REPORT.txt files in scene code directories")
            print("  3. Consider simplifying complex animations")
            print("  4. Manually edit and retry failed scenes")
            print("-"*70)
        else:
            print("\n🎉 All scenes rendered successfully!")
        
        print("="*70 + "\n")
        
        # Clear failed scenes for next run
        self.failed_scenes = []

    def run_manim_process(self, topic: str):
        """
        Run manim on all generated manim code for a specific topic using VideoRenderer.

        Args:
            topic (str): The topic to render videos for
        """
        print(f"[{topic}] ==> Manually running Manim process for all scenes...")
        result = self.video_renderer.run_manim_process(topic)
        print(f"[{topic}] <== Manim process finished.")
        return result

    def create_snapshot_scene(
        self,
        topic: str,
        scene_number: int,
        version_number: int,
        return_type: str = "image",
    ):
        """
        Create a snapshot of the video for a specific topic and scene using VideoRenderer.

        Args:
            topic (str): The topic of the video
            scene_number (int): Scene number to snapshot
            version_number (int): Version number to snapshot
            return_type (str, optional): Type of snapshot to return. Defaults to "image".

        Returns:
            The snapshot in the specified format
        """
        print(
            f"[{topic} | Scene {scene_number}] ==> Creating snapshot for version {version_number}..."
        )
        snapshot = self.video_renderer.create_snapshot_scene(
            topic, scene_number, version_number, return_type
        )
        print(f"[{topic} | Scene {scene_number}] <== Snapshot created.")
        return snapshot

    def _find_latest_video_for_scene(
        self, project_path: str, scene_num: int
    ) -> str | None:
        # Look for videos in the media/videos/ subdirectory
        videos_dir = os.path.join(project_path, "media", "videos")
        search_pattern = os.path.join(videos_dir, f"*scene{scene_num}_v*")
        potential_folders = glob.glob(search_pattern)
        if not potential_folders:
            return None
        latest_folder = max(
            potential_folders,
            key=lambda p: (
                int(re.search(r"_v(\d+)", p).group(1))
                if re.search(r"_v(\d+)", p)
                else -1
            ),
        )
        for res in ["1080p60", "720p30", "480p15"]:
            video_file = os.path.join(latest_folder, res, f"Scene{scene_num}.mp4")
            if os.path.exists(video_file):
                return video_file
        return None

    def combine_videos(self, topic: str):
        """
        Combines all videos for a topic and generates the final, fine-grained
        Temporal Context Map (TCM) by scaling Proto-TCMs.
        """
        file_prefix = topic.lower().replace(" ", "_")
        project_path = os.path.join(
            self.output_dir, file_prefix, file_prefix
        )  # Adjust if your path structure is different
        project_name = topic
        inner_folder_name = os.path.basename(project_path)

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
            scene_num = int(
                re.search(r"scene(\d+)", os.path.basename(scene_dir)).group(1)
            )
            video_path = self._find_latest_video_for_scene(project_path, scene_num)
            proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")

            if not video_path or not os.path.exists(
                os.path.join(scene_dir, "succ_rendered.txt")
            ):
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
                    scaled_duration = (
                        event.get("estimatedDuration", 1.0) * scaling_factor
                    )
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
                    f"  - Processed Scene {scene_num} (Duration: {actual_duration:.2f}s), created {len(proto_tcm)} TCM entries."
                )
                global_time_offset += actual_duration

            except Exception as e:
                print(f"  - ERROR processing Scene {scene_num}: {e}")

        if video_clips_paths:
            clips = [VideoFileClip(p) for p in video_clips_paths]
            final_video_clip = concatenate_videoclips(clips)
            output_video_path = os.path.join(
                project_path, f"{inner_folder_name}_combined.mp4"
            )
            output_tcm_path = os.path.join(
                project_path, f"{inner_folder_name}_combined_tcm.json"
            )

            final_video_clip.write_videofile(
                output_video_path, codec="libx264", audio_codec="aac", logger="bar"
            )
            with open(output_tcm_path, "w", encoding="utf-8") as f:
                json.dump(final_tcm, f, indent=2, ensure_ascii=False)

            print(
                f"[{topic}] ==> Project finalized. Video and fine-grained TCM created."
            )
            for clip in clips:
                clip.close()
            final_video_clip.close()
        else:
            print(f"[{topic}] <== No rendered scenes found to finalize.")

    async def _generate_scene_implementation_single(
        self,
        topic: str,
        description: str,
        scene_outline_i: str,
        i: int,
        file_prefix: str,
        session_id: str,
        scene_trace_id: str,
    ) -> dict:
        """
        Orchestrates the generation of a detailed plan and Proto-TCM for a single scene.
        Supports both legacy LLM text responses and structured dict responses.
        """
        print(
            f"[{topic} | Scene {i}] ---> Creating implementation plan and Proto-TCM..."
        )

        # Call the planner
        full_llm_response_obj = (
            await self.planner._generate_scene_implementation_single(
                topic,
                description,
                scene_outline_i,
                i,
                file_prefix,
                session_id,
                scene_trace_id,
            )
        )

        # --- NEW: Handle structured planner dict responses directly ---
        if (
            isinstance(full_llm_response_obj, dict)
            and "plan" in full_llm_response_obj
            and "proto_tcm" in full_llm_response_obj
        ):
            plan = full_llm_response_obj["plan"]
            proto_tcm_str = full_llm_response_obj["proto_tcm"]
        else:
            # --- Legacy handling (raw string or OpenAI-style dict) ---
            full_llm_response = ""

            if isinstance(full_llm_response_obj, str):
                full_llm_response = full_llm_response_obj
            elif isinstance(full_llm_response_obj, dict):
                try:
                    # Standard OpenAI / LiteLLM format
                    full_llm_response = full_llm_response_obj["choices"][0]["message"][
                        "content"
                    ]
                except (KeyError, IndexError, TypeError):
                    # Fallback for {'content': '...'}
                    if "content" in full_llm_response_obj:
                        full_llm_response = full_llm_response_obj["content"]
                    else:
                        print(
                            f"[{topic} | Scene {i}] WARNING: Could not determine content key in LLM response dict. Using string representation."
                        )
                        full_llm_response = str(full_llm_response_obj)
            else:
                print(
                    f"[{topic} | Scene {i}] WARNING: Received unexpected type from planner: {type(full_llm_response_obj)}. Converting to string."
                )
                full_llm_response = str(full_llm_response_obj)

            # Extract sections from XML-like response
            plan = extract_xml(full_llm_response, "SCENE_TECHNICAL_IMPLEMENTATION_PLAN")
            if not plan or "<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>" not in plan:
                # Fallback if tag missing
                plan = full_llm_response

            proto_tcm_str = extract_xml(full_llm_response, "SCENE_PROTO_TCM")

        # --- Save artifacts ---
        scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{i}")
        os.makedirs(scene_dir, exist_ok=True)

        # Save Proto-TCM if valid JSON
        if proto_tcm_str and "<SCENE_PROTO_TCM>" not in proto_tcm_str:
            try:
                proto_tcm_data = json.loads(proto_tcm_str)
                proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")
                with open(proto_tcm_path, "w", encoding="utf-8") as f:
                    json.dump(proto_tcm_data, f, indent=2, ensure_ascii=False)
                print(f"[{topic} | Scene {i}] <--- Proto-TCM saved successfully.")
            except json.JSONDecodeError:
                print(
                    f"[{topic} | Scene {i}] WARNING: Failed to parse Proto-TCM JSON from planner response."
                )
                proto_tcm_str = ""  # Invalidate if not proper JSON
        else:
            print(
                f"[{topic} | Scene {i}] WARNING: Proto-TCM block not found in planner response."
            )
            proto_tcm_str = ""

        # Save the implementation plan
        plan_path = os.path.join(
            scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt"
        )
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan)

        # Return structured dict
        return {"plan": plan, "proto_tcm": proto_tcm_str}

    async def generate_video_pipeline(
        self,
        topic: str,
        description: str,
        max_retries: int,
        only_plan: bool = False,
        specific_scenes: List[int] = None,
        progress_callback=None,
    ):
        """
        Modified pipeline to handle partial scene completions and option to only generate plans for specific scenes.
        This version processes each scene sequentially: code generation followed immediately by rendering.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            max_retries (int): Maximum number of code fix attempts
            only_plan (bool, optional): Whether to only generate plans without rendering. Defaults to False.
            specific_scenes (List[int], optional): List of specific scenes to process. Defaults to None.
        """
        print(f"======================================================")
        print(f"🚀 STARTING VIDEO PIPELINE FOR TOPIC: {topic}")
        print(f"======================================================")

        # Load or create session ID for the topic folder
        topic_folder_session_file = os.path.join(self.output_dir, topic, "session_id.txt")
        if os.path.exists(topic_folder_session_file):
            with open(topic_folder_session_file, "r") as f:
                session_id = f.read().strip()
                # print(f"Loaded existing topic session ID: {session_id}")
        else:
            session_id = self._load_or_create_session_id()
            # Save session_id in the topic folder (output/Topic Name/session_id.txt)
            os.makedirs(os.path.join(self.output_dir, topic), exist_ok=True)
            with open(topic_folder_session_file, "w", encoding='utf-8') as f:
                f.write(session_id)
            print(f"Saved topic session ID to {topic_folder_session_file}")
        
        # Also save in the project folder (output/Topic Name/topic_name/session_id.txt)
        self._save_topic_session_id(topic, session_id)

        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)

        # PHASE 1: SCENE OUTLINE
        # Create structure: output/Topic Name/topic_name/
        scene_outline_path = os.path.join(
            self.output_dir, topic, file_prefix, f"{file_prefix}_scene_outline.txt"
        )
        print("\n[PHASE 1: SCENE OUTLINE]")
        if progress_callback:
            progress_callback(0.05, "📝 Planning your video structure...")
        if os.path.exists(scene_outline_path):
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            print(f"[{topic}] Loaded existing scene outline.")
        else:
            print(f"[{topic}] No scene outline found. Generating a new one...")
            # The planner returns a full response object/dictionary, not just a string.
            scene_outline_obj = self.planner.generate_scene_outline(
                topic, description, session_id
            )

            # Safely extract the string content from the response object.
            scene_outline = ""
            if isinstance(scene_outline_obj, str):
                scene_outline = scene_outline_obj
            elif isinstance(scene_outline_obj, dict):
                try:
                    # Standard accessor for LiteLLM/OpenAI responses
                    scene_outline = scene_outline_obj["choices"][0]["message"][
                        "content"
                    ]
                except (KeyError, IndexError, TypeError):
                    # Fallback for a simpler {'content': '...'} structure
                    if "content" in scene_outline_obj:
                        scene_outline = scene_outline_obj["content"]
                    else:
                        print(
                            f"[{topic}] WARNING: Could not determine content key in scene_outline dict. Using string representation."
                        )
                        scene_outline = str(scene_outline_obj)
            else:
                print(
                    f"[{topic}] WARNING: Received unexpected type for scene_outline: {type(scene_outline_obj)}. Converting to string."
                )
                scene_outline = str(scene_outline_obj)
            if not scene_outline or "<SCENE_OUTLINE>" not in scene_outline:
                raise ValueError(
                    f"[{topic}] FAILED to generate a valid scene outline. Aborting."
                )
            os.makedirs(os.path.join(self.output_dir, topic, file_prefix), exist_ok=True)
            with open(scene_outline_path, "w", encoding="utf-8") as f:
                f.write(scene_outline)
            print(f"[{topic}] New scene outline saved.")
        print("[PHASE 1 COMPLETE]")
        if progress_callback:
            progress_callback(0.15, "✅ Video structure ready")

        # CLEANUP: Remove invalid success markers
        print("\n[CLEANUP: VALIDATING SUCCESS MARKERS]")
        removed = self.cleanup_invalid_success_markers(topic)
        if removed > 0:
            print(f"[{topic}] Removed {removed} invalid succ_rendered.txt file(s)")
        
        # PRE-FLIGHT CHECK: Ensure all scene plans are ready
        print("\n[PRE-FLIGHT CHECK: VALIDATING SCENE PLANS]")
        if progress_callback:
            progress_callback(0.17, "🔍 Validating existing plans...")
        
        plans_ready = await self.ensure_all_scene_plans_ready(
            topic, description, scene_outline, session_id
        )
        
        if not plans_ready:
            print(f"\n⚠️  WARNING: Not all scene plans could be generated.")
            print(f"   This may be due to API rate limits or other issues.")
            print(f"   Continuing with available plans...")
        
        print("[PRE-FLIGHT CHECK COMPLETE]")
        if progress_callback:
            progress_callback(0.19, "✅ Plans validated")

        # PHASE 2: IMPLEMENTATION PLANS (PLAN ALL)
        print("\n[PHASE 2: IMPLEMENTATION PLANS]")
        if progress_callback:
            progress_callback(0.20, "🎨 Designing each scene...")
        implementation_plans_dict = self.load_implementation_plans(topic)
        scene_outline_content = extract_xml(scene_outline, "SCENE_OUTLINE")
        scene_numbers = (
            len(re.findall(r"<SCENE_(\d+)>[^<]", scene_outline_content))
            if scene_outline_content
            else 0
        )

        missing_scenes = []
        completed_plans = []
        for i in range(1, scene_numbers + 1):
            if implementation_plans_dict.get(i) is None and (
                specific_scenes is None or i in specific_scenes
            ):
                missing_scenes.append(i)
            elif implementation_plans_dict.get(i) is not None:
                completed_plans.append(i)
        
        # Report plan status
        if completed_plans:
            print(f"[{topic}] ✅ Scenes with plans ready: {completed_plans}")
            if progress_callback:
                progress_callback(0.22, f"✅ {len(completed_plans)} of {scene_numbers} scenes already planned")

        if missing_scenes:
            print(
                f"[{topic}] Generating missing implementation plans for scenes: {missing_scenes}"
            )
            if progress_callback:
                progress_callback(0.23, f"🎨 Planning {len(missing_scenes)} remaining scenes...")
            for idx, scene_num in enumerate(missing_scenes):
                if progress_callback:
                    plan_progress = 0.20 + (0.15 * (idx / len(missing_scenes)))
                    progress_callback(plan_progress, f"🎨 Designing scene {scene_num} of {scene_numbers}...")
                scene_match = re.search(
                    f"<SCENE_{scene_num}>(.*?)</SCENE_{scene_num}>",
                    scene_outline_content,
                    re.DOTALL,
                )
                if scene_match:
                    scene_outline_i = scene_match.group(1)
                    scene_trace_id = str(uuid.uuid4())
                    implementation_details = (
                        await self._generate_scene_implementation_single(
                            topic,
                            description,
                            scene_outline_i,
                            scene_num,
                            file_prefix,
                            session_id,
                            scene_trace_id,
                        )
                    )
                    implementation_plans_dict[scene_num] = implementation_details[
                        "plan"
                    ]
            print(f"[{topic}] Finished generating missing plans.")
        else:
            print(f"[{topic}] All required implementation plans are present.")
        print("[PHASE 2 COMPLETE]")
        if progress_callback:
            progress_callback(0.35, "✅ All scenes designed")

        if only_plan:
            print(
                f"\n[PIPELINE STOP] 'only_plan' is True. Skipping code generation and rendering."
            )
            return

        # PHASE 3: SEQUENTIAL CODE GENERATION & RENDERING
        print(f"\n[PHASE 3: CODE GENERATION & RENDERING (SCENE-BY-SCENE)]")
        if progress_callback:
            progress_callback(0.40, "🎬 Creating animations...")
        sorted_scene_numbers = sorted(implementation_plans_dict.keys())
        
        # Count scenes to process and already completed
        scenes_to_process = [s for s in sorted_scene_numbers if not specific_scenes or s in specific_scenes]
        total_scenes_to_process = len(scenes_to_process)
        
        # Check which scenes are already rendered
        completed_scenes = []
        pending_scenes = []
        for scene_num in sorted_scene_numbers:
            scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{scene_num}")
            is_rendered = os.path.exists(os.path.join(scene_dir, "succ_rendered.txt"))
            if is_rendered and not args.only_render:
                completed_scenes.append(scene_num)
            else:
                pending_scenes.append(scene_num)
        
        # Report status
        if completed_scenes:
            print(f"[{topic}] ✅ Already completed scenes: {completed_scenes}")
            if progress_callback:
                progress_callback(0.40, f"✅ {len(completed_scenes)} of {scene_numbers} scenes complete")
        if pending_scenes:
            print(f"[{topic}] 🎬 Scenes to render: {pending_scenes}")
            if progress_callback:
                progress_callback(0.42, f"🎬 Starting code generation for {len(pending_scenes)} scenes...")
        
        processed_count = 0

        for scene_num in sorted_scene_numbers:
            # Check if this scene should be processed
            if specific_scenes and scene_num not in specific_scenes:
                if self.verbose:
                    print(
                        f"[{topic} | Scene {scene_num}] Skipping, not in specified list."
                    )
                continue

            scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{scene_num}")
            is_rendered = os.path.exists(os.path.join(scene_dir, "succ_rendered.txt"))

            if is_rendered and not args.only_render:
                print(f"[{topic} | Scene {scene_num}] ✅ Already rendered, skipping.")
                continue

            print(f"\n--- Processing Scene {scene_num} ---")
            implementation_plan = implementation_plans_dict.get(scene_num)
            if not implementation_plan:
                print(
                    f"[{topic} | Scene {scene_num}] WARNING: Skipping scene as its implementation plan is missing."
                )
                continue

            # Load Proto-TCM for the scene
            proto_tcm_str = ""
            proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")
            if os.path.exists(proto_tcm_path):
                with open(proto_tcm_path, "r") as f:
                    proto_tcm_str = f.read()

            # Get or create scene_trace_id
            scene_trace_id_path = os.path.join(
                scene_dir, "subplans", "scene_trace_id.txt"
            )
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
                i=scene_num - 1,
                scene_outline=scene_outline,
                scene_implementation=implementation_plan,
                proto_tcm=proto_tcm_str,
                topic=topic,
                description=description,
                max_retries=max_retries,
                file_prefix=file_prefix,
                session_id=session_id,
                scene_trace_id=scene_trace_id,
            )
            
            processed_count += 1
            if progress_callback:
                scene_progress = 0.40 + (0.50 * (processed_count / total_scenes_to_process))
                progress_callback(scene_progress, f"✅ Scene {scene_num} done!")
        
        print("\n[PHASE 3 COMPLETE]")
        if progress_callback:
            progress_callback(0.90, "✅ All animations complete!")

        if not args.only_render:
            print(
                f"\n[FINALIZING] Video rendering pipeline completed for topic '{topic}'."
            )
        else:
            print(
                f"\n[PIPELINE STOP] 'only_render' is True. Skipping final combination."
            )

        print(f"======================================================")
        print(f"✅ PIPELINE FINISHED FOR TOPIC: {topic}")
        print(f"======================================================")

    def check_theorem_status(self, theorem: Dict) -> Dict[str, bool]:
        """
        Check if a theorem has its plan, code files, and rendered videos with detailed scene status.

        Args:
            theorem (Dict): Dictionary containing theorem information

        Returns:
            Dict[str, bool]: Dictionary containing status information for the theorem
        """
        topic = theorem["theorem"]
        if self.verbose:
            print(f"Checking status for topic: {topic}...")
        file_prefix = topic.lower()
        file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)

        # Check scene outline
        scene_outline_path = os.path.join(
            self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt"
        )
        has_scene_outline = os.path.exists(scene_outline_path)

        # Get number of scenes if outline exists
        num_scenes = 0
        if has_scene_outline:
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            scene_outline_content = extract_xml(scene_outline, "SCENE_OUTLINE")
            if scene_outline_content:
                num_scenes = len(
                    re.findall(r"<SCENE_(\d+)>[^<]", scene_outline_content)
                )

        # Check implementation plans, code files, and rendered videos
        implementation_plans = 0
        code_files = 0
        rendered_scenes = 0

        # Track status of individual scenes
        scene_status = []
        for i in range(1, num_scenes + 1):
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i}")

            # Check implementation plan
            plan_path = os.path.join(
                scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt"
            )
            has_plan = os.path.exists(plan_path)
            if has_plan:
                implementation_plans += 1

            # Check code files
            code_dir = os.path.join(scene_dir, "code")
            has_code = False
            if os.path.exists(code_dir):
                if any(f.endswith(".py") for f in os.listdir(code_dir)):
                    has_code = True
                    code_files += 1

            # Check rendered scene video
            has_render = False
            succ_rendered_path = os.path.join(scene_dir, "succ_rendered.txt")
            if os.path.exists(succ_rendered_path):
                has_render = True
                rendered_scenes += 1

            scene_status.append(
                {
                    "scene_number": i,
                    "has_plan": has_plan,
                    "has_code": has_code,
                    "has_render": has_render,
                }
            )

        # Check combined video
        combined_video_path = os.path.join(
            self.output_dir, file_prefix, f"{file_prefix}_combined.mp4"
        )
        has_combined_video = os.path.exists(combined_video_path)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Manim videos using AI")
    parser.add_argument(
        "--model",
        type=str,
        choices=allowed_models,
        default="gemini/gemini-1.5-pro-002",
        help="Select the AI model to use",
    )
    parser.add_argument(
        "--topic", type=str, default=None, help="Topic to generate videos for"
    )
    parser.add_argument(
        "--context", type=str, default=None, help="Context of the topic"
    )
    parser.add_argument(
        "--only_gen_vid",
        action="store_true",
        help="Only generate videos to existing plans",
    )
    parser.add_argument(
        "--only_combine", action="store_true", help="Only combine videos"
    )
    parser.add_argument(
        "--peek_existing_videos",
        "--peek",
        action="store_true",
        help="Peek at existing videos",
    )
    parser.add_argument(
        "--output_dir", type=str, default=Config.OUTPUT_DIR, help="Output directory"
    )  # Use Config
    parser.add_argument(
        "--theorems_path", type=str, default=None, help="Path to theorems json file"
    )
    parser.add_argument(
        "--sample_size",
        "--sample",
        type=int,
        default=None,
        help="Number of theorems to sample",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of retries for code generation",
    )
    parser.add_argument(
        "--use_rag",
        "--rag",
        action="store_true",
        help="Use Retrieval Augmented Generation",
    )
    parser.add_argument(
        "--use_visual_fix_code",
        "--visual_fix_code",
        action="store_true",
        help="Use VLM to fix code with rendered visuals",
    )
    parser.add_argument(
        "--chroma_db_path",
        type=str,
        default=Config.CHROMA_DB_PATH,
        help="Path to Chroma DB",
    )  # Use Config
    parser.add_argument(
        "--manim_docs_path",
        type=str,
        default=Config.MANIM_DOCS_PATH,
        help="Path to manim docs",
    )  # Use Config
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=Config.EMBEDDING_MODEL,  # Use Config
        choices=["azure/text-embedding-3-large", "vertex_ai/text-embedding-005"],
        help="Select the embedding model to use",
    )
    parser.add_argument(
        "--use_context_learning",
        action="store_true",
        help="Use context learning with example Manim code",
    )
    parser.add_argument(
        "--context_learning_path",
        type=str,
        default=Config.CONTEXT_LEARNING_PATH,  # Use Config
        help="Path to context learning examples",
    )
    parser.add_argument(
        "--use_langfuse", action="store_true", help="Enable Langfuse logging"
    )
    parser.add_argument(
        "--max_scene_concurrency",
        type=int,
        default=7,
        help="Maximum number of scenes to process concurrently",
    )
    parser.add_argument(
        "--max_topic_concurrency",
        type=int,
        default=7,
        help="Maximum number of topics to process concurrently",
    )
    parser.add_argument(
        "--debug_combine_topic", type=str, help="Debug combine videos", default=None
    )
    parser.add_argument(
        "--only_plan",
        action="store_true",
        help="Only generate scene outline and implementation plans",
    )
    parser.add_argument(
        "--check_status",
        action="store_true",
        help="Check planning and code status for all theorems",
    )
    parser.add_argument(
        "--only_render",
        action="store_true",
        help="Only render scenes without combining videos",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=int,
        help="Specific scenes to process (if theorems_path is provided)",
    )
    args = parser.parse_args()

    # --- Start of Execution ---
    print("=============================================")
    print("======= AI Manim Video Generator CLI ========")
    print("=============================================")
    print("\nCONFIGURATION:")
    for arg, value in vars(args).items():
        print(f"  - {arg}: {value}")
    print("\n")

    # Initialize planner model using LiteLLM
    if args.verbose:
        verbose = True
    else:
        verbose = False

    print("Initializing LLM Wrappers...")
    planner_model = LiteLLMWrapper(
        model_name=args.model,
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse,
    )
    scene_model = LiteLLMWrapper(  # Initialize scene_model separately
        model_name=args.model,
        temperature=0.7,
        print_cost=True,
        verbose=verbose,
        use_langfuse=args.use_langfuse,
    )
    print(f"Models Initialized -> Planner: {args.model}, Scene: {args.model}")

    if args.theorems_path:
        print(f"Loading theorems from: {args.theorems_path}")
        with open(args.theorems_path, "r", encoding="utf-8") as f:
            theorems = json.load(f)

        if args.sample_size:
            print(f"Sampling {args.sample_size} theorems from the list.")
            theorems = theorems[: args.sample_size]

        if args.peek_existing_videos:
            print(f"\n[MODE: PEEK EXISTING VIDEOS in {args.output_dir}]")
            successful_rendered_videos = 0
            total_folders = 0
            for item in os.listdir(args.output_dir):
                if os.path.isdir(os.path.join(args.output_dir, item)):
                    total_folders += 1
                    if os.path.exists(
                        os.path.join(args.output_dir, item, f"{item}_combined.mp4")
                    ):
                        successful_rendered_videos += 1
            print(
                f"Combined Video Status: {successful_rendered_videos}/{total_folders} completed."
            )

            successful_rendered_scenes = 0
            total_scenes = 0
            for item in os.listdir(args.output_dir):
                if os.path.isdir(os.path.join(args.output_dir, item)):
                    for scene_folder in os.listdir(os.path.join(args.output_dir, item)):
                        if "scene" in scene_folder and os.path.isdir(
                            os.path.join(args.output_dir, item, scene_folder)
                        ):
                            total_scenes += 1
                            if os.path.exists(
                                os.path.join(
                                    args.output_dir,
                                    item,
                                    scene_folder,
                                    "succ_rendered.txt",
                                )
                            ):
                                successful_rendered_scenes += 1
            print(
                f"Individual Scene Status: {successful_rendered_scenes}/{total_scenes} successfully rendered."
            )
            exit()

        video_generator = VideoGenerator(
            planner_model=planner_model,
            scene_model=scene_model,
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_rag=args.use_rag,
            use_context_learning=args.use_context_learning,
            context_learning_path=args.context_learning_path,
            chroma_db_path=args.chroma_db_path,
            manim_docs_path=args.manim_docs_path,
            embedding_model=args.embedding_model,
            use_visual_fix_code=args.use_visual_fix_code,
            use_langfuse=args.use_langfuse,
            max_scene_concurrency=args.max_scene_concurrency,
        )

        if args.debug_combine_topic is not None:
            print(f"\n[MODE: DEBUG COMBINE for topic: {args.debug_combine_topic}]")
            video_generator.combine_videos(args.debug_combine_topic)
            exit()

        if args.only_gen_vid:
            print(f"\n[MODE: GENERATE VIDEOS ONLY from existing plans]")

            async def process_theorem(theorem, topic_semaphore):
                async with topic_semaphore:
                    topic = theorem["theorem"]
                    description = theorem["description"]
                    print(f"--- Starting video generation for topic: {topic} ---")

                    file_prefix = topic.lower()
                    file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)
                    scene_outline_path = os.path.join(
                        video_generator.output_dir,
                        file_prefix,
                        f"{file_prefix}_scene_outline.txt",
                    )

                    if not os.path.exists(scene_outline_path):
                        print(f"Skipping topic '{topic}': Scene outline not found.")
                        return

                    with open(scene_outline_path, "r") as f:
                        scene_outline = f.read()

                    implementation_plans_dict = (
                        video_generator.load_implementation_plans(topic)
                    )
                    implementation_plans = [
                        (num, plan)
                        for num, plan in implementation_plans_dict.items()
                        if plan is not None
                    ]

                    if not implementation_plans:
                        print(
                            f"Skipping topic '{topic}': No implementation plans found."
                        )
                        return

                    session_id = (
                        video_generator._load_topic_session_id(topic)
                        or video_generator.session_id
                    )

                    await video_generator.render_video_fix_code(
                        topic=topic,
                        description=description,
                        scene_outline=scene_outline,
                        implementation_plans=implementation_plans,
                        max_retries=args.max_retries,
                        session_id=session_id,
                    )
                    print(f"--- Finished video generation for topic: {topic} ---")

            async def main():
                print(
                    f"Starting concurrent processing of {len(theorems)} theorems with max topic concurrency: {args.max_topic_concurrency}"
                )
                topic_semaphore = asyncio.Semaphore(args.max_topic_concurrency)
                tasks = [
                    process_theorem(theorem, topic_semaphore) for theorem in theorems
                ]
                await asyncio.gather(*tasks)

            asyncio.run(main())

        elif args.check_status:
            print("\n[MODE: CHECK STATUS for all theorems]")
            video_generator = VideoGenerator(
                planner_model=planner_model,
                scene_model=scene_model,
                output_dir=args.output_dir,
                verbose=args.verbose,
                use_rag=args.use_rag,
                use_context_learning=args.use_context_learning,
                context_learning_path=args.context_learning_path,
                chroma_db_path=args.chroma_db_path,
                manim_docs_path=args.manim_docs_path,
                embedding_model=args.embedding_model,
                use_visual_fix_code=args.use_visual_fix_code,
                use_langfuse=args.use_langfuse,
                max_scene_concurrency=args.max_scene_concurrency,
            )

            all_statuses = [
                video_generator.check_theorem_status(theorem) for theorem in theorems
            ]

            # Print combined status table
            print("\n" + "=" * 160)
            print("THEOREM STATUS OVERVIEW")
            print("=" * 160)
            print(
                f"{'Topic':<40} {'Outline':<8} {'Total':<8} {'Status (P=Plan, C=Code, R=Render)':<50} {'Combined':<10} {'Missing Components':<40}"
            )
            print("-" * 160)
            for status in all_statuses:
                scene_status_str = ""
                for scene in status["scene_status"]:
                    scene_str = (
                        ("P" if scene["has_plan"] else "-")
                        + ("C" if scene["has_code"] else "-")
                        + ("R" if scene["has_render"] else "-")
                        + " "
                    )
                    scene_status_str += scene_str

                missing_plans = [
                    str(s["scene_number"])
                    for s in status["scene_status"]
                    if not s["has_plan"]
                ]
                missing_code = [
                    str(s["scene_number"])
                    for s in status["scene_status"]
                    if not s["has_code"]
                ]
                missing_renders = [
                    str(s["scene_number"])
                    for s in status["scene_status"]
                    if not s["has_render"]
                ]

                missing_str = []
                if missing_plans:
                    missing_str.append(f"P:{','.join(missing_plans)}")
                if missing_code:
                    missing_str.append(f"C:{','.join(missing_code)}")
                if missing_renders:
                    missing_str.append(f"R:{','.join(missing_renders)}")
                missing_str = " ".join(missing_str)

                print(
                    f"{status['topic'][:37]+'...' if len(status['topic'])>37 else status['topic']:<40} "
                    f"{'✓' if status['has_scene_outline'] else '✗':<8} "
                    f"{status['total_scenes']:<8} "
                    f"{scene_status_str[:47]+'...' if len(scene_status_str)>47 else scene_status_str:<50} "
                    f"{'✓' if status['has_combined_video'] else '✗':<10} "
                    f"{missing_str[:37]+'...' if len(missing_str)>37 else missing_str:<40}"
                )

            print("-" * 160)
            print("\nSUMMARY:")
            total_theorems = len(theorems)
            total_scenes_overall = sum(s["total_scenes"] for s in all_statuses)
            total_plans = sum(s["implementation_plans"] for s in all_statuses)
            total_codes = sum(s["code_files"] for s in all_statuses)
            total_renders = sum(s["rendered_scenes"] for s in all_statuses)
            total_combined = sum(1 for s in all_statuses if s["has_combined_video"])

            print(f"  - Total Theorems Checked: {total_theorems}")
            print(f"  - Total Scenes Overall: {total_scenes_overall}")
            print(
                f"  - Scene Plan Completion: {total_plans}/{total_scenes_overall:.1% if total_scenes_overall else 0.0}%"
            )
            print(
                f"  - Scene Code Completion: {total_codes}/{total_scenes_overall:.1% if total_scenes_overall else 0.0}%"
            )
            print(
                f"  - Scene Render Completion: {total_renders}/{total_scenes_overall:.1% if total_scenes_overall else 0.0}%"
            )
            print(
                f"  - Combined Video Completion: {total_combined}/{total_theorems:.1% if total_theorems else 0.0}%"
            )
            exit()

        else:
            print("\n[MODE: FULL PIPELINE (from scratch or resume)]")

            async def process_theorem(theorem, topic_semaphore, index, total):
                async with topic_semaphore:
                    topic = theorem["theorem"]
                    description = theorem["description"]
                    print(f"\n>>> Processing Theorem {index+1}/{total}: {topic}")
                    try:
                        if args.only_combine:
                            video_generator.combine_videos(topic)
                        else:
                            await video_generator.generate_video_pipeline(
                                topic,
                                description,
                                max_retries=args.max_retries,
                                only_plan=args.only_plan,
                                specific_scenes=args.scenes,
                            )
                            if not args.only_plan and not args.only_render:
                                video_generator.combine_videos(topic)
                    except Exception as e:
                        print(f"FATAL ERROR processing topic '{topic}': {e}")
                        # Optionally, log this error to a file
                    finally:
                        print(f"<<< Finished Theorem {index+1}/{total}: {topic}\n")

            async def main():
                print(
                    f"Starting concurrent processing of {len(theorems)} theorems with max topic concurrency: {args.max_topic_concurrency}"
                )
                topic_semaphore = asyncio.Semaphore(args.max_topic_concurrency)
                tasks = [
                    process_theorem(theorem, topic_semaphore, i, len(theorems))
                    for i, theorem in enumerate(theorems)
                ]
                await asyncio.gather(*tasks)

            asyncio.run(main())

    elif args.topic and args.context:
        print("\n[MODE: SINGLE TOPIC PROCESSING]")
        video_generator = VideoGenerator(
            planner_model=planner_model,
            scene_model=scene_model,
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_rag=args.use_rag,
            use_context_learning=args.use_context_learning,
            context_learning_path=args.context_learning_path,
            chroma_db_path=args.chroma_db_path,
            manim_docs_path=args.manim_docs_path,
            embedding_model=args.embedding_model,
            use_visual_fix_code=args.use_visual_fix_code,
            use_langfuse=args.use_langfuse,
            max_scene_concurrency=args.max_scene_concurrency,
        )
        print(f"Processing single topic: {args.topic}")

        if args.only_gen_vid:
            print("Sub-mode: Generating video for existing plans...")
            file_prefix = args.topic.lower()
            file_prefix = re.sub(r"[^a-z0-9_]+", "_", file_prefix)
            topic_dir = os.path.join(video_generator.output_dir, file_prefix)
            scene_outline_path = os.path.join(
                topic_dir, f"{file_prefix}_scene_outline.txt"
            )

            if not os.path.exists(scene_outline_path):
                print(
                    f"FATAL ERROR: Scene outline not found at '{scene_outline_path}'. Cannot generate video without plans."
                )
                exit(1)

            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()

            implementation_plans_dict = video_generator.load_implementation_plans(
                args.topic
            )
            implementation_plans = [
                (num, plan)
                for num, plan in implementation_plans_dict.items()
                if plan is not None
            ]

            if not implementation_plans:
                print(
                    f"FATAL ERROR: No implementation plans found for topic '{args.topic}'."
                )
                exit(1)

            session_id = (
                video_generator._load_topic_session_id(args.topic)
                or video_generator.session_id
            )

            asyncio.run(
                video_generator.render_video_fix_code(
                    topic=args.topic,
                    description=args.context,
                    scene_outline=scene_outline,
                    implementation_plans=implementation_plans,
                    max_retries=args.max_retries,
                    session_id=session_id,
                )
            )
            print("Video generation for existing plans complete.")
            exit()

        if args.only_combine:
            print("Sub-mode: Combining videos only...")
            video_generator.combine_videos(args.topic)
        else:
            print("Sub-mode: Running full pipeline...")
            try:
                asyncio.run(
                    video_generator.generate_video_pipeline(
                        args.topic,
                        args.context,
                        max_retries=args.max_retries,
                        only_plan=args.only_plan,
                    )
                )
                if not args.only_plan and not args.only_render:
                    video_generator.combine_videos(args.topic)
            except Exception as e:
                print(f"FATAL ERROR during pipeline for topic '{args.topic}': {e}")
                exit(1)
    else:
        print("\nERROR: Invalid arguments.")
        print(
            "Please provide either (--theorems_path) or both (--topic and --context)."
        )
        exit(1)

    print("\n=============================================")
    print("========== All Tasks Completed! ===========")
    print("=============================================")

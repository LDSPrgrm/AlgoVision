"""
Error recovery utilities for handling LLM rate limits and corrupted files.

This module provides functions to detect and recover from LLM API errors
by identifying corrupted plan files and cleaning them up for regeneration.
"""

import os
import re
import glob
from typing import List, Dict, Tuple


class ErrorRecovery:
    """Handles detection and recovery from LLM errors in generated files."""
    
    # Common error patterns that indicate LLM failures
    ERROR_PATTERNS = [
        # Rate limit errors
        r"rate.?limit",
        r"quota.?exceeded",
        r"too.?many.?requests",
        r"429",
        r"resource.?exhausted",
        
        # Overload errors
        r"overloaded",
        r"capacity",
        r"temporarily.?unavailable",
        r"503",
        r"502",
        r"service.?unavailable",
        
        # Timeout errors
        r"timeout",
        r"timed.?out",
        r"deadline.?exceeded",
        r"gateway.?timeout",
        r"504",
        
        # API errors
        r"api.?error",
        r"internal.?server.?error",
        r"500",
        r"bad.?gateway",
        
        # Incomplete responses
        r"^$",  # Empty file
        r"^\s*$",  # Only whitespace
        r"^Error:",  # Starts with Error
        r"^Failed to",  # Starts with Failed
        r"^Exception:",  # Starts with Exception
        
        # Malformed XML/JSON - Opening tags without content
        r"<SCENE_VISION_STORYBOARD_PLAN>\s*$",
        r"<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>\s*$",
        r"<SCENE_ANIMATION_NARRATION_PLAN>\s*$",
        r"<SCENE_OUTLINE>\s*$",
        r"^\s*\{?\s*$",  # Empty JSON
        r"^\[\s*\]$",  # Empty JSON array
    ]
    
    # File patterns for different plan types
    PLAN_FILE_PATTERNS = {
        'vision_storyboard': '*_vision_storyboard_plan.txt',
        'technical_implementation': '*_technical_implementation_plan.txt',
        'animation_narration': '*_animation_narration_plan.txt',
        'implementation_plan': '*_implementation_plan.txt',
        'proto_tcm': 'proto_tcm.json',
        'scene_outline': '*_scene_outline.txt',
    }
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize ErrorRecovery.
        
        Args:
            output_dir (str): Base output directory for generated files
        """
        self.output_dir = output_dir
        self.recently_cleaned = {}  # Track recently cleaned files to avoid re-cleaning
    
    def validate_plan_structure(self, file_path: str, content: str) -> Tuple[bool, str]:
        """
        Validate the structure of plan files based on expected XML tags and bracket sections.
        
        Args:
            file_path (str): Path to the file being validated
            content (str): Content of the file
            
        Returns:
            Tuple[bool, str]: (is_valid, error_description)
        """
        # Vision Storyboard Plan validation
        if '_vision_storyboard_plan.txt' in file_path:
            required_tags = [
                '<SCENE_VISION_STORYBOARD_PLAN>',
                '</SCENE_VISION_STORYBOARD_PLAN>',
                '[SCENE_VISION]',
                '[STORYBOARD]'
            ]
            for tag in required_tags:
                if tag not in content:
                    return False, f"missing_required_tag: {tag}"
        
        # Technical Implementation Plan validation
        elif '_technical_implementation_plan.txt' in file_path:
            required_tags = [
                '<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>',
                '</SCENE_TECHNICAL_IMPLEMENTATION_PLAN>'
            ]
            for tag in required_tags:
                if tag not in content:
                    return False, f"missing_required_tag: {tag}"
        
        # Animation Narration Plan validation
        elif '_animation_narration_plan.txt' in file_path:
            required_tags = [
                '<SCENE_ANIMATION_NARRATION_PLAN>',
                '</SCENE_ANIMATION_NARRATION_PLAN>',
                '[ANIMATION_STRATEGY]',
                '[NARRATION]'
            ]
            for tag in required_tags:
                if tag not in content:
                    return False, f"missing_required_tag: {tag}"
        
        # Scene Outline validation
        elif '_scene_outline.txt' in file_path:
            required_tags = [
                '<SCENE_OUTLINE>',
                '</SCENE_OUTLINE>'
            ]
            for tag in required_tags:
                if tag not in content:
                    return False, f"missing_required_tag: {tag}"
            
            # Check for at least one scene
            if not re.search(r'<SCENE_\d+>', content):
                return False, "no_scenes_defined"
            
            # Check that scenes have closing tags
            scene_numbers = re.findall(r'<SCENE_(\d+)>', content)
            for num in scene_numbers:
                if f'</SCENE_{num}>' not in content:
                    return False, f"scene_{num}_missing_closing_tag"
        
        # Implementation Plan (combined) validation
        elif '_implementation_plan.txt' in file_path and 'scene' in file_path:
            # Should contain all three sub-plans
            required_tags = [
                '<SCENE_VISION_STORYBOARD_PLAN>',
                '<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>',
                '<SCENE_ANIMATION_NARRATION_PLAN>'
            ]
            found_tags = sum(1 for tag in required_tags if tag in content)
            if found_tags < 2:  # At least 2 out of 3 should be present
                return False, f"combined_plan_incomplete (found {found_tags}/3 sections)"
        
        return True, ""
    
    def is_file_corrupted(self, file_path: str) -> Tuple[bool, str]:
        """
        Check if a file contains LLM error patterns.
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            Tuple[bool, str]: (is_corrupted, error_type)
        """
        if not os.path.exists(file_path):
            return False, ""
        
        # Check if this file was recently cleaned (within last 10 minutes)
        # to avoid re-detecting a file that's being regenerated
        import time
        if file_path in self.recently_cleaned:
            time_since_clean = time.time() - self.recently_cleaned[file_path]
            if time_since_clean < 600:  # 10 minutes (increased from 5)
                # File is being regenerated, don't check yet
                return False, ""
        
        # Also check file modification time - if file was just created/modified
        # in the last 2 minutes, skip checking (it's likely being written)
        try:
            file_mtime = os.path.getmtime(file_path)
            time_since_modified = time.time() - file_mtime
            if time_since_modified < 120:  # 2 minutes
                # File was just created/modified, skip checking
                return False, ""
        except OSError:
            pass  # File doesn't exist or can't get mtime
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file is too small (likely incomplete)
            if len(content.strip()) < 50:
                return True, "incomplete_response"
            
            # Check for error patterns
            content_lower = content.lower()
            for pattern in self.ERROR_PATTERNS:
                if re.search(pattern, content_lower, re.IGNORECASE | re.MULTILINE):
                    return True, f"error_pattern: {pattern}"
            
            # Check for malformed XML tags (opening without closing)
            xml_tags = [
                'SCENE_VISION_STORYBOARD_PLAN',
                'SCENE_TECHNICAL_IMPLEMENTATION_PLAN',
                'SCENE_ANIMATION_NARRATION_PLAN',
                'SCENE_OUTLINE',
                'SCENE_VISION',
                'STORYBOARD',
                'ANIMATION_STRATEGY',
                'NARRATION'
            ]
            for tag in xml_tags:
                opening = f"<{tag}>"
                closing = f"</{tag}>"
                if opening in content and closing not in content:
                    return True, f"malformed_xml_missing_closing: {tag}"
                # Also check for closing without opening
                if closing in content and opening not in content:
                    return True, f"malformed_xml_missing_opening: {tag}"
            

            
            # Check for malformed JSON in proto_tcm
            if file_path.endswith('proto_tcm.json'):
                import json
                try:
                    data = json.loads(content)
                    # Check if it's empty or has no events
                    if not data or (isinstance(data, list) and len(data) == 0):
                        return True, "empty_proto_tcm"
                    # Check if it's a list with valid event structure
                    if isinstance(data, list):
                        for event in data:
                            if not isinstance(event, dict):
                                return True, "invalid_proto_tcm_structure"
                            # Check for required fields
                            if 'conceptName' not in event and 'narrationText' not in event:
                                return True, "proto_tcm_missing_required_fields"
                except json.JSONDecodeError:
                    return True, "invalid_json"
            
            # Validate plan file structure
            is_valid, error_desc = self.validate_plan_structure(file_path, content)
            if not is_valid:
                return True, error_desc
            
            return False, ""
            
        except Exception as e:
            print(f"Error checking file {file_path}: {e}")
            return False, ""
    
    def scan_scene_for_errors(self, topic: str, scene_number: int) -> Dict[str, List[str]]:
        """
        Scan all files for a specific scene to detect errors.
        
        Args:
            topic (str): Topic name
            scene_number (int): Scene number to scan
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping error types to list of corrupted files
        """
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        scene_dir = os.path.join(self.output_dir, topic, file_prefix, f"scene{scene_number}")
        subplan_dir = os.path.join(scene_dir, "subplans")
        
        corrupted_files = {}
        
        # Check subplan files
        if os.path.exists(subplan_dir):
            for plan_type, pattern in self.PLAN_FILE_PATTERNS.items():
                if plan_type in ['implementation_plan', 'scene_outline']:
                    continue  # These are in parent directory
                
                files = glob.glob(os.path.join(subplan_dir, pattern))
                for file_path in files:
                    is_corrupted, error_type = self.is_file_corrupted(file_path)
                    if is_corrupted:
                        if error_type not in corrupted_files:
                            corrupted_files[error_type] = []
                        corrupted_files[error_type].append(file_path)
        
        # Check scene-level files
        for plan_type in ['implementation_plan', 'proto_tcm']:
            pattern = self.PLAN_FILE_PATTERNS[plan_type]
            files = glob.glob(os.path.join(scene_dir, pattern))
            for file_path in files:
                is_corrupted, error_type = self.is_file_corrupted(file_path)
                if is_corrupted:
                    if error_type not in corrupted_files:
                        corrupted_files[error_type] = []
                    corrupted_files[error_type].append(file_path)
        
        return corrupted_files
    
    def scan_topic_for_errors(self, topic: str) -> Dict[int, Dict[str, List[str]]]:
        """
        Scan all scenes in a topic for errors.
        
        Args:
            topic (str): Topic name
            
        Returns:
            Dict[int, Dict[str, List[str]]]: Dictionary mapping scene numbers to their corrupted files
        """
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        topic_dir = os.path.join(self.output_dir, topic, file_prefix)
        
        if not os.path.exists(topic_dir):
            return {}
        
        # Find all scene directories
        scene_dirs = glob.glob(os.path.join(topic_dir, "scene*"))
        scene_errors = {}
        
        for scene_dir in scene_dirs:
            scene_match = re.search(r'scene(\d+)', os.path.basename(scene_dir))
            if scene_match:
                scene_num = int(scene_match.group(1))
                errors = self.scan_scene_for_errors(topic, scene_num)
                if errors:
                    scene_errors[scene_num] = errors
        
        # Also check scene outline
        outline_path = os.path.join(topic_dir, f"{file_prefix}_scene_outline.txt")
        if os.path.exists(outline_path):
            is_corrupted, error_type = self.is_file_corrupted(outline_path)
            if is_corrupted:
                scene_errors[0] = {error_type: [outline_path]}  # Use 0 for outline
        
        return scene_errors
    
    def clean_corrupted_files(self, topic: str, scene_number: int = None, 
                             dry_run: bool = True) -> List[str]:
        """
        Delete corrupted files to allow regeneration.
        
        Args:
            topic (str): Topic name
            scene_number (int, optional): Specific scene to clean, or None for all scenes
            dry_run (bool): If True, only report what would be deleted without deleting
            
        Returns:
            List[str]: List of files that were (or would be) deleted
        """
        import time
        deleted_files = []
        
        if scene_number is not None:
            # Clean specific scene
            errors = self.scan_scene_for_errors(topic, scene_number)
            for error_type, files in errors.items():
                for file_path in files:
                    deleted_files.append(file_path)
                    if not dry_run:
                        os.remove(file_path)
                        # Track that we cleaned this file
                        self.recently_cleaned[file_path] = time.time()
                        print(f"Deleted corrupted file: {file_path} (Error: {error_type})")
                    else:
                        print(f"Would delete: {file_path} (Error: {error_type})")
        else:
            # Clean all scenes
            all_errors = self.scan_topic_for_errors(topic)
            for scene_num, errors in all_errors.items():
                scene_label = "Scene Outline" if scene_num == 0 else f"Scene {scene_num}"
                print(f"\n{scene_label}:")
                for error_type, files in errors.items():
                    for file_path in files:
                        deleted_files.append(file_path)
                        if not dry_run:
                            os.remove(file_path)
                            # Track that we cleaned this file
                            self.recently_cleaned[file_path] = time.time()
                            print(f"  Deleted: {os.path.basename(file_path)} (Error: {error_type})")
                        else:
                            print(f"  Would delete: {os.path.basename(file_path)} (Error: {error_type})")
        
        return deleted_files
    
    def auto_recover_scene(self, topic: str, scene_number: int) -> bool:
        """
        Automatically detect and clean corrupted files for a scene.
        
        Args:
            topic (str): Topic name
            scene_number (int): Scene number
            
        Returns:
            bool: True if any files were cleaned, False otherwise
        """
        print(f"[Auto-Recovery] Scanning {topic} Scene {scene_number} for errors...")
        errors = self.scan_scene_for_errors(topic, scene_number)
        
        if not errors:
            print(f"[Auto-Recovery] No errors detected in Scene {scene_number}")
            return False
        
        print(f"[Auto-Recovery] Found errors in Scene {scene_number}:")
        for error_type, files in errors.items():
            print(f"  - {error_type}: {len(files)} file(s)")
        
        # Clean the corrupted files
        deleted = self.clean_corrupted_files(topic, scene_number, dry_run=False)
        print(f"[Auto-Recovery] Cleaned {len(deleted)} corrupted file(s)")
        print(f"[Auto-Recovery] Scene {scene_number} ready for regeneration")
        
        return True
    
    def generate_recovery_report(self, topic: str) -> str:
        """
        Generate a detailed report of all errors in a topic.
        
        Args:
            topic (str): Topic name
            
        Returns:
            str: Formatted report
        """
        all_errors = self.scan_topic_for_errors(topic)
        
        if not all_errors:
            return f"✓ No errors detected in topic '{topic}'"
        
        report = []
        report.append("="*70)
        report.append(f"ERROR RECOVERY REPORT: {topic}")
        report.append("="*70)
        report.append(f"Total scenes with errors: {len(all_errors)}")
        report.append("")
        
        for scene_num, errors in sorted(all_errors.items()):
            scene_label = "Scene Outline" if scene_num == 0 else f"Scene {scene_num}"
            report.append(f"\n{scene_label}:")
            report.append("-"*70)
            
            total_files = sum(len(files) for files in errors.values())
            report.append(f"  Corrupted files: {total_files}")
            
            for error_type, files in errors.items():
                report.append(f"\n  Error Type: {error_type}")
                for file_path in files:
                    report.append(f"    - {os.path.basename(file_path)}")
        
        report.append("\n" + "="*70)
        report.append("RECOMMENDATIONS:")
        report.append("  1. Run clean_corrupted_files() to remove bad files")
        report.append("  2. Regenerate the affected scenes")
        report.append("  3. Check API rate limits and quotas")
        report.append("="*70)
        
        return "\n".join(report)


# Convenience functions
def scan_and_report(topic: str, output_dir: str = "output") -> str:
    """Quick scan and report for a topic."""
    recovery = ErrorRecovery(output_dir)
    return recovery.generate_recovery_report(topic)


def auto_clean_topic(topic: str, output_dir: str = "output", dry_run: bool = True) -> int:
    """Automatically clean all corrupted files in a topic."""
    recovery = ErrorRecovery(output_dir)
    deleted = recovery.clean_corrupted_files(topic, dry_run=dry_run)
    return len(deleted)

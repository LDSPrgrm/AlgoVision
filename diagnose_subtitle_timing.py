#!/usr/bin/env python3
"""
Diagnostic tool to check subtitle timing issues.
Run this to analyze a project's subtitle timing.
"""

import json
import os
import sys
import glob
import re

def diagnose_project(project_name):
    """Diagnose subtitle timing for a project."""
    
    # Find project path
    output_dir = "output"
    project_path = None
    
    for top_level_dir in os.listdir(output_dir):
        if top_level_dir == project_name:
            top_level_path = os.path.join(output_dir, top_level_dir)
            for inner_item in os.listdir(top_level_path):
                inner_path = os.path.join(top_level_path, inner_item)
                if os.path.isdir(inner_path):
                    project_path = inner_path
                    break
            break
    
    if not project_path:
        print(f"❌ Project '{project_name}' not found")
        return
    
    print(f"📁 Project: {project_name}")
    print(f"📂 Path: {project_path}")
    print("=" * 80)
    
    # Check combined TCM
    inner_folder_name = os.path.basename(project_path)
    tcm_path = os.path.join(project_path, f"{inner_folder_name}_combined_tcm.json")
    
    if not os.path.exists(tcm_path):
        print("❌ No combined TCM found. Generate video first.")
        return
    
    with open(tcm_path, "r", encoding="utf-8") as f:
        tcm_data = json.load(f)
    
    print(f"\n📊 TCM Analysis:")
    print(f"Total events: {len(tcm_data)}")
    
    # Check for sync fix marker
    has_sync_fix = any(event.get("_sync_fix_applied", False) for event in tcm_data)
    print(f"Sync fix applied: {'✅ Yes' if has_sync_fix else '❌ No (regenerate needed!)'}")
    
    # Analyze events
    events_with_narration = 0
    events_without_narration = 0
    zero_duration_events = 0
    total_duration = 0.0
    
    print(f"\n📝 Event Breakdown:")
    print(f"{'Scene':<8} {'Concept':<30} {'Narration':<15} {'Duration':<10} {'Time Range'}")
    print("-" * 80)
    
    for event in tcm_data:
        narration = event.get("narrationText", "").strip()
        start = float(event.get("startTime", 0))
        end = float(event.get("endTime", 0))
        duration = end - start
        concept = event.get("conceptName", "unknown")[:28]
        
        # Extract scene number from conceptId
        concept_id = event.get("conceptId", "")
        scene_match = re.search(r'scene_(\d+)', concept_id)
        scene_num = scene_match.group(1) if scene_match else "?"
        
        has_narration = narration and narration != "..."
        
        if has_narration:
            events_with_narration += 1
            narration_display = "Yes"
        else:
            events_without_narration += 1
            narration_display = "No (empty)"
        
        if duration == 0:
            zero_duration_events += 1
        
        total_duration = max(total_duration, end)
        
        # Show first 20 events
        if events_with_narration + events_without_narration <= 20:
            print(f"Scene {scene_num:<3} {concept:<30} {narration_display:<15} {duration:>6.2f}s    {start:.2f}s → {end:.2f}s")
    
    if len(tcm_data) > 20:
        print(f"... ({len(tcm_data) - 20} more events)")
    
    print("\n" + "=" * 80)
    print(f"\n📈 Summary:")
    print(f"Events with narration: {events_with_narration}")
    print(f"Events without narration: {events_without_narration}")
    print(f"Zero-duration events: {zero_duration_events}")
    print(f"Total timeline duration: {total_duration:.2f}s")
    
    # Check for issues
    print(f"\n🔍 Potential Issues:")
    
    if not has_sync_fix:
        print("⚠️  Sync fix NOT applied - regenerate subtitles!")
    
    if events_without_narration > 0 and zero_duration_events != events_without_narration:
        print(f"⚠️  {events_without_narration} events without narration, but only {zero_duration_events} have 0 duration")
        print("    This could cause subtitle drift!")
    
    # Check scene proto_tcm files
    print(f"\n🎬 Scene Analysis:")
    scene_dirs = sorted(
        glob.glob(os.path.join(project_path, "scene*")),
        key=lambda x: int(re.search(r"scene(\d+)", x).group(1)) if re.search(r"scene(\d+)", x) else 0
    )
    
    for scene_dir in scene_dirs:
        scene_num = re.search(r"scene(\d+)", os.path.basename(scene_dir))
        if not scene_num:
            continue
        scene_num = scene_num.group(1)
        
        proto_tcm_path = os.path.join(scene_dir, "proto_tcm.json")
        if not os.path.exists(proto_tcm_path):
            print(f"  Scene {scene_num}: ❌ No proto_tcm.json")
            continue
        
        with open(proto_tcm_path, "r", encoding="utf-8") as f:
            proto_tcm = json.load(f)
        
        # Check for voiceover cache
        cache_dirs = [
            os.path.join(scene_dir, "code", "media", "voiceovers"),
            os.path.join(project_path, "media", "voiceovers"),
        ]
        
        has_cache = False
        audio_count = 0
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                json_files = glob.glob(os.path.join(cache_dir, "*.json"))
                audio_count = len(json_files)
                if audio_count > 0:
                    has_cache = True
                    break
        
        narration_count = sum(1 for e in proto_tcm if e.get("narrationText", "").strip() and e.get("narrationText", "").strip() != "...")
        
        cache_status = f"✅ {audio_count} audio files" if has_cache else "❌ No cache"
        print(f"  Scene {scene_num}: {len(proto_tcm)} events, {narration_count} with narration, {cache_status}")
    
    print("\n" + "=" * 80)
    print("\n💡 Recommendations:")
    
    if not has_sync_fix:
        print("1. Regenerate subtitles using the Utilities tab")
    elif events_without_narration > 0 and zero_duration_events != events_without_narration:
        print("1. Regenerate subtitles - some empty events have incorrect duration")
    else:
        print("1. Subtitles look correctly configured")
        print("2. If still lagging, the issue might be:")
        print("   - Video player not syncing properly")
        print("   - Audio encoding delay in the video file")
        print("   - Proto-TCM events don't match actual audio")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_subtitle_timing.py <project_name>")
        print("\nExample: python diagnose_subtitle_timing.py 'Merge Sort'")
        sys.exit(1)
    
    project_name = sys.argv[1]
    diagnose_project(project_name)

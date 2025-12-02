#!/usr/bin/env python3
"""
Check audio synchronization in the combined video.
This script analyzes the audio track to find when actual speech starts.
"""

from moviepy import VideoFileClip, AudioFileClip
import numpy as np
import json

def find_audio_start(audio_array, sample_rate, threshold=0.01):
    """Find when audio actually starts (above threshold)."""
    # Calculate RMS energy in small windows
    window_size = int(sample_rate * 0.1)  # 100ms windows
    for i in range(0, len(audio_array) - window_size, window_size):
        window = audio_array[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        if rms > threshold:
            return i / sample_rate
    return 0.0

def analyze_video_audio_sync(video_path, tcm_path):
    """Analyze if video audio matches TCM timing."""
    print(f"Analyzing: {video_path}")
    print("=" * 80)
    
    # Load video and extract audio
    with VideoFileClip(video_path) as video:
        print(f"Video duration: {video.duration:.3f}s")
        
        if video.audio is None:
            print("ERROR: No audio track found!")
            return
        
        # Get audio as numpy array
        audio_array = video.audio.to_soundarray(fps=22050)
        if len(audio_array.shape) > 1:
            # Convert stereo to mono
            audio_array = audio_array.mean(axis=1)
        
        sample_rate = 22050
        
        # Find when audio actually starts
        audio_start = find_audio_start(audio_array, sample_rate)
        print(f"Audio starts at: {audio_start:.3f}s")
        
        if audio_start > 0.1:
            print(f"⚠️  WARNING: {audio_start:.3f}s of silence at the beginning!")
            print(f"   This will cause subtitles to appear {audio_start:.3f}s too early")
    
    # Load TCM
    with open(tcm_path, 'r', encoding='utf-8') as f:
        tcm = json.load(f)
    
    print(f"\nTCM Analysis:")
    print(f"Total events: {len(tcm)}")
    print(f"First event starts at: {tcm[0]['startTime']}s")
    print(f"First event text: {tcm[0]['narrationText'][:60]}...")
    print(f"Last event ends at: {tcm[-1]['endTime']}s")
    
    # Check for gaps in TCM
    print(f"\nChecking for timing gaps in TCM:")
    prev_end = 0.0
    gaps_found = 0
    for i, event in enumerate(tcm[:10]):  # Check first 10 events
        start = float(event['startTime'])
        end = float(event['endTime'])
        gap = start - prev_end
        if gap > 0.01:  # More than 10ms gap
            print(f"  Event {i}: Gap of {gap:.3f}s before this event")
            gaps_found += 1
        prev_end = end
    
    if gaps_found == 0:
        print("  ✓ No significant gaps found in first 10 events")
    
    # Calculate total narration time
    total_narration_time = sum(
        float(e['endTime']) - float(e['startTime'])
        for e in tcm
        if e.get('narrationText', '').strip() and e.get('narrationText', '').strip() != '...'
    )
    print(f"\nTotal narration time: {total_narration_time:.3f}s")
    print(f"Video duration: {video.duration:.3f}s")
    print(f"Difference: {video.duration - total_narration_time:.3f}s")
    
    if audio_start > 0.1:
        print(f"\n🎯 ROOT CAUSE IDENTIFIED:")
        print(f"   Video has {audio_start:.3f}s of silence at the start")
        print(f"   Subtitles start at 0.000s but audio starts at {audio_start:.3f}s")
        print(f"   Solution: Add {audio_start:.3f}s offset to all subtitle times")
    else:
        print(f"\n✓ No significant audio delay detected")
        print(f"  If subtitles still lag, the issue might be:")
        print(f"  1. Video player subtitle rendering delay")
        print(f"  2. Audio encoding delay")
        print(f"  3. Individual audio file timing issues")

if __name__ == "__main__":
    video_path = "output/Merge Sort/merge_sort/merge_sort_combined.mp4"
    tcm_path = "output/Merge Sort/merge_sort/merge_sort_combined_tcm.json"
    
    try:
        analyze_video_audio_sync(video_path, tcm_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

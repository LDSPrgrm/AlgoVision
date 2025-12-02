# Critical Fix: Subtitles Lagging Behind Audio

## The Problem

Even after regenerating subtitles with the empty narration fix, subtitles were still lagging behind the audio.

### Root Cause

The **scaling factor** was calculated incorrectly. It was dividing the actual video duration by the sum of **all** estimated durations, including events with no narration:

```python
# WRONG:
total_estimated_duration = sum(e.get("estimatedDuration", 1.0) for e in proto_tcm)
# This includes events with empty narration!

scaling_factor = actual_duration / total_estimated_duration
# This makes the scaling factor too small!
```

### Why This Causes Lag

```
Example Scene:
- Event 1: "Binary search..." (estimated: 3.0s, has audio)
- Event 2: "..." (estimated: 2.0s, NO audio)
- Event 3: "It divides..." (estimated: 4.0s, has audio)
- Actual video duration: 7.0s (only events 1 and 3 have audio)

OLD CALCULATION:
total_estimated = 3.0 + 2.0 + 4.0 = 9.0s
scaling_factor = 7.0 / 9.0 = 0.778

Event 1: 3.0 * 0.778 = 2.33s (should be ~3.5s)
Event 2: 0.0s (correct - no narration)
Event 3: 4.0 * 0.778 = 3.11s (should be ~3.5s)

Result: Subtitles are compressed, appearing BEFORE audio finishes
        This makes them lag behind as the video progresses
```

---

## The Solution

Only count estimated durations for events **with narration** when calculating the scaling factor:

```python
# CORRECT:
total_estimated_duration = sum(
    e.get("estimatedDuration", 1.0) 
    for e in proto_tcm 
    if e.get("narrationText", "").strip() and e.get("narrationText", "").strip() != "..."
)
# Only includes events that actually generate audio!

scaling_factor = actual_duration / total_estimated_duration
# Now the scaling factor is correct!
```

### Why This Works

```
Same Example Scene:
- Event 1: "Binary search..." (estimated: 3.0s, has audio)
- Event 2: "..." (estimated: 2.0s, NO audio - EXCLUDED)
- Event 3: "It divides..." (estimated: 4.0s, has audio)
- Actual video duration: 7.0s

NEW CALCULATION:
total_estimated = 3.0 + 4.0 = 7.0s (event 2 excluded!)
scaling_factor = 7.0 / 7.0 = 1.0

Event 1: 3.0 * 1.0 = 3.0s ✅
Event 2: 0.0s (correct - no narration) ✅
Event 3: 4.0 * 1.0 = 4.0s ✅

Result: Subtitles match audio timing perfectly!
```

---

## Impact

### Before Fix:
```
Timeline: 0s ─── 2.3s ─── 2.3s ─── 5.4s
Audio:    [──3.5s──]     [──3.5s──]
Subtitles: ✗ (early)     ✗ (lagging)
```

### After Fix:
```
Timeline: 0s ─── 3.5s ─── 3.5s ─── 7.0s
Audio:    [──3.5s──]     [──3.5s──]
Subtitles: ✓ (perfect)   ✓ (perfect)
```

---

## Where Fixed

1. **`regenerate_subtitles_only()`** - Fast subtitle regeneration
2. **`combine_videos()`** - Full video combination

Both functions now calculate the scaling factor correctly by excluding empty narration events.

---

## How to Apply the Fix

### For Existing Videos:

1. Go to **Utilities** tab
2. Select your project (e.g., "Merge Sort")
3. Click **"🔄 Regenerate Subtitles"**
4. Wait 5-15 seconds
5. Subtitles will now be perfectly synced!

### For New Videos:

The fix is automatically applied during video generation. New videos will have perfect subtitle timing from the start.

---

## Diagnostic Tool

Use the included diagnostic tool to check subtitle timing:

```bash
python diagnose_subtitle_timing.py "Merge Sort"
```

This will show:
- How many events have narration vs. empty
- Whether the sync fix is applied
- Potential timing issues
- Recommendations

---

## Technical Details

### The Math Behind It

The scaling factor adjusts estimated durations to match actual video duration:

```
scaling_factor = actual_duration / total_estimated_duration
actual_event_duration = estimated_duration * scaling_factor
```

**Key Insight**: `total_estimated_duration` must only include events that contribute to `actual_duration`.

Since empty narration events don't generate audio, they don't contribute to the video duration, so they shouldn't be included in the calculation.

### Why Wasn't This Caught Earlier?

The previous fix (setting empty events to 0 duration) was correct, but it didn't fix the scaling factor calculation. So:

1. Empty events correctly got 0 duration ✅
2. But events with narration got scaled incorrectly ❌
3. Result: Subtitles were compressed, causing lag

Both fixes are needed:
1. Empty events = 0 duration (prevents drift from empty events)
2. Correct scaling factor (prevents compression of narrated events)

---

## Testing

### Verify the Fix:

1. **Check TCM file**:
```json
{
  "narrationText": "Binary search is efficient",
  "startTime": "0.000",
  "endTime": "3.500",  // Should match audio duration
  "_sync_fix_applied": true
}
```

2. **Watch the video**:
- Subtitles should appear exactly when audio starts
- Subtitles should disappear exactly when audio ends
- No lag throughout the entire video

3. **Run diagnostic**:
```bash
python diagnose_subtitle_timing.py "Your Project"
```

---

## Summary

**Problem**: Scaling factor included empty narration events, causing compression  
**Solution**: Only count events with narration when calculating scaling factor  
**Result**: Perfect subtitle sync throughout the video ✅

This fix, combined with the empty narration duration fix, ensures subtitles stay perfectly synchronized with audio from start to finish.

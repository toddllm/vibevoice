#!/usr/bin/env python3
"""
Test script for Tier 1 implementation
"""

import requests
import json
import base64
import time

def test_synthesis(script, duration, quality="balanced"):
    """Test the synthesis endpoint"""
    print(f"\n📝 Testing: {len(script)} chars, {duration}s, quality={quality}")
    
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:5001/synthesize",
        json={
            "script": script,
            "target_seconds": duration,
            "quality": quality
        },
        headers={"Content-Type": "application/json"}
    )
    
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        metrics = data['metrics']
        
        # Decode and save WAV
        wav_data = base64.b64decode(data['wav_b64'])
        filename = f"test_{duration}s_{quality}.wav"
        with open(filename, "wb") as f:
            f.write(wav_data)
        
        print(f"✅ Success in {elapsed:.1f}s")
        print(f"   Generated: {metrics['seconds_generated']:.1f}s")
        print(f"   RTF: {metrics['rtf']:.2f}x")
        print(f"   Frames: {metrics['frames_emitted']}/{metrics['frames_total']}")
        print(f"   Saved: {filename} ({len(wav_data)} bytes)")
        return True
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return False

def check_status():
    """Check server status"""
    response = requests.get("http://localhost:5001/status")
    if response.status_code == 200:
        data = response.json()
        print("📊 Server Status:")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Device: {data['device']}")
        if data['memory_model']:
            mm = data['memory_model']
            print(f"   Memory: {mm['safe_available_gb']:.1f}GB available")
            print(f"   Max safe duration: ~{mm['max_safe_frames']/7.5:.1f}s")
        if data['limits']:
            print(f"   Max text tokens: {data['limits']['max_text_tokens']}")
            print(f"   Max audio minutes: {data['limits']['max_audio_minutes']}")
        return True
    return False

if __name__ == "__main__":
    print("🧪 Testing VibeVoice Tier 1 Implementation")
    
    # Check status
    if not check_status():
        print("❌ Server not responding. Start with: python vibevoice_tier1.py")
        exit(1)
    
    # Test cases
    short_text = "Hello, this is a test of the VibeVoice system."
    
    medium_text = """
    Welcome to this demonstration of VibeVoice, a state-of-the-art text-to-speech system.
    The model can generate natural sounding speech with proper intonation and pacing.
    It handles various punctuation marks, including commas, periods, and questions.
    Isn't that impressive?
    """
    
    long_text = """
    This is a longer test to demonstrate extended generation capabilities.
    VibeVoice uses a novel architecture with acoustic frames operating at 7.5 Hz,
    which provides an efficient representation of audio while maintaining high quality.
    The system can handle multiple sentences and maintain consistency throughout the generation.
    With proper memory management and duration control, we can generate several minutes of audio
    without running into memory issues. The real-time factor shows how fast the generation is
    compared to the duration of the generated audio. A factor above 1.0 means faster than real-time.
    """
    
    # Run tests
    print("\n" + "="*60)
    
    # Test 1: Short, fast quality
    test_synthesis(short_text, 10, "fast")
    
    # Test 2: Medium, balanced quality  
    test_synthesis(medium_text, 30, "balanced")
    
    # Test 3: Long, high quality
    test_synthesis(long_text, 60, "high")
    
    print("\n" + "="*60)
    print("✅ All tests complete! Check the generated .wav files")
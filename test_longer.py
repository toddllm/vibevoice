#!/usr/bin/env python3
"""
Test longer text generation with VibeVoice
"""

import requests
import json

def test_generation(text, duration):
    """Test TTS generation with specified duration"""
    print(f"Testing with {len(text)} characters, max duration: {duration}s")
    
    response = requests.post(
        "http://localhost:5000/generate",
        json={
            "text": text,
            "max_duration": duration
        },
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        # Save audio
        filename = f"test_output_{duration}s.wav"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Generated {len(response.content)} bytes -> {filename}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

# Test with longer text
long_text = """
Hello! This is a test of the VibeVoice text-to-speech system with longer content generation. 
We're testing the ability to generate speech for extended periods of time. 
The model should be able to produce natural-sounding speech that maintains consistency 
throughout the entire duration. This is particularly useful for creating podcasts, 
audiobooks, and other long-form audio content. Let's see how well it handles 
multiple sentences with various punctuation marks, pauses, and different types of content. 
The quality should remain high even as the generation continues for several minutes.
"""

if __name__ == "__main__":
    print("Testing VibeVoice with different durations...")
    
    # Test short duration
    test_generation("This is a short test.", 10)
    
    # Test medium duration  
    test_generation(long_text[:200], 30)
    
    # Test longer duration
    test_generation(long_text, 60)
    
    print("\nTest complete! Check the generated .wav files")
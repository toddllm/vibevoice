#!/usr/bin/env python3
"""
Test Tier 2 chunking and multi-speaker synthesis
"""

import requests
import json
import base64
import time

def test_chunked_synthesis():
    """Test chunked synthesis with multi-speaker text"""
    
    # Multi-speaker dialogue
    multi_speaker_text = """
Speaker 0: Welcome to our technology podcast. Today we're discussing the latest advances in AI.
Speaker 1: Thank you for having me. It's an exciting time to be in this field.
Speaker 0: Let's start with text-to-speech. The quality has improved dramatically. What do you think are the key innovations?
Speaker 1: Well, I think the biggest breakthrough has been the use of neural architectures that can capture long-range dependencies. Models like VibeVoice can now generate speech for up to 90 minutes while maintaining consistency.
Speaker 0: That's remarkable. How does it handle multiple speakers?
Speaker 1: The model can handle up to four distinct speakers, each with their own voice characteristics. It maintains speaker identity throughout the conversation.
Speaker 0: What about the technical challenges?
Speaker 1: Memory management is crucial. The model operates at 7.5 Hz for acoustic frames, which is very efficient. This allows for real-time or faster generation on modern hardware.
    """
    
    # Single long text (should trigger multiple chunks)
    long_single_text = """
    This is a comprehensive test of the chunking system. The text will be automatically divided into manageable chunks based on the frame budget. Each chunk will be generated separately, then crossfaded together to create seamless audio. The system handles natural endpointing, where the model may stop generation when it detects the end of an utterance. This is actually beneficial, as it creates more natural speech patterns. The budget controller ensures we generate enough audio to meet the target duration, starting new chunks as needed. The crossfade algorithm uses equal-power blending with an overlap of about one second, creating smooth transitions between chunks. Peak normalization is applied to each chunk before crossfading to ensure consistent volume levels. The entire process is designed to be robust and handle various edge cases, including very short chunks, speaker changes, and early termination. By embracing the model's natural behavior rather than fighting it, we achieve better quality and more reliable generation.
    """
    
    print("="*60)
    print("Testing Tier 2 Chunked Synthesis")
    print("="*60)
    
    tests = [
        ("Multi-speaker dialogue", multi_speaker_text, 60),
        ("Long single speaker", long_single_text, 45),
    ]
    
    for test_name, text, duration in tests:
        print(f"\n📝 Test: {test_name}")
        print(f"   Duration: {duration}s, Text length: {len(text)} chars")
        
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5002/synthesize_chunked",
            json={
                "script": text,
                "target_seconds": duration,
                "quality": "balanced"
            },
            headers={"Content-Type": "application/json"}
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Analyze events
            chunk_events = [e for e in data['events'] if e['type'] == 'chunk_done']
            done_event = next((e for e in data['events'] if e['type'] == 'done'), None)
            
            print(f"✅ Success in {elapsed:.1f}s")
            print(f"   Chunks processed: {len(chunk_events)}")
            
            # Show chunk details
            for i, event in enumerate(chunk_events):
                print(f"   Chunk {i+1}: {event['chunk_frames']} frames, "
                      f"speaker: {event.get('speaker', 'unknown')}, "
                      f"RTF: {event.get('rtf', 0):.2f}x")
            
            if done_event:
                print(f"   Total: {done_event['total_seconds']:.1f}s generated, "
                      f"RTF: {done_event['rtf']:.2f}x")
                
                # Save audio
                if 'wav_b64' in data:
                    wav_data = base64.b64decode(data['wav_b64'])
                    filename = f"test_tier2_{test_name.replace(' ', '_')}.wav"
                    with open(filename, 'wb') as f:
                        f.write(wav_data)
                    print(f"   Saved: {filename} ({len(wav_data)} bytes)")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

def test_sse_stream():
    """Test SSE streaming (requires manual observation)"""
    print("\n" + "="*60)
    print("Testing SSE Stream")
    print("="*60)
    
    print("ℹ️  SSE testing requires browser or specialized client")
    print("   Open http://localhost:5002 in browser to test SSE")
    print("   Or use curl: curl -N -H 'Content-Type: application/json' \\")
    print("                -d '{\"script\":\"Test text\", \"target_seconds\":10}' \\")
    print("                http://localhost:5002/synthesize_stream")

if __name__ == "__main__":
    print("🧪 Testing VibeVoice Tier 2 Implementation")
    
    # Check if server is running
    try:
        r = requests.get("http://localhost:5002/")
        print("✅ Server is running")
    except:
        print("❌ Server not responding. Start with: python vibevoice_tier2.py")
        exit(1)
    
    # Run tests
    test_chunked_synthesis()
    test_sse_stream()
    
    print("\n" + "="*60)
    print("✅ Tests complete! Check generated .wav files")
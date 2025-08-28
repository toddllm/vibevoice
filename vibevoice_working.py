#!/usr/bin/env python3
"""
Working VibeVoice TTS implementation with proper AudioStreamer and threading
"""

import threading
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import time

# Import VibeVoice components
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

def synthesize_to_wav(text, output_path="output.wav", speaker_voice_path="demo/voices/en-Alice_woman.wav"):
    """
    Generate speech from text using VibeVoice with proper streaming
    """
    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"🔧 Using device: {device}, dtype: {dtype}")
    
    # Load model
    print("📥 Loading VibeVoice model...")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-1.5B",
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",  # Use eager since flash_attention_2 has issues
    )
    model.eval()
    
    # Load processor
    processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
    print("✅ Model loaded successfully!")
    
    # Load voice sample
    print(f"🎤 Loading voice sample: {speaker_voice_path}")
    wav, sr = sf.read(speaker_voice_path)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    if sr != 24000:
        print(f"  Resampling from {sr}Hz to 24000Hz...")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
    print(f"  Voice sample shape: {wav.shape}")
    
    # Format text with speaker label
    formatted_text = f"Speaker 0: {text}"
    print(f"📝 Text: {formatted_text}")
    
    # Prepare inputs
    inputs = processor(
        text=[formatted_text],
        voice_samples=[[wav]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Create AudioStreamer
    audio_streamer = AudioStreamer(
        batch_size=1,
        stop_signal=None,
        timeout=None
    )
    
    # Variable to store any exception from generation thread
    generation_exception = None
    
    def run_generation():
        """Run generation in background thread"""
        nonlocal generation_exception
        try:
            print("🚀 Starting generation in background thread...")
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    tokenizer=processor.tokenizer,
                    audio_streamer=audio_streamer,  # Critical!
                    max_new_tokens=500,  # Adjust as needed
                    do_sample=False,  # Deterministic for testing
                    cfg_scale=1.3,
                    generation_config={
                        'do_sample': False,
                    }
                )
            print("✅ Generation complete!")
        except Exception as e:
            generation_exception = e
            print(f"❌ Generation error: {e}")
        finally:
            # Always signal end of stream
            audio_streamer.end()
    
    # Start generation in background thread
    generation_thread = threading.Thread(target=run_generation)
    generation_thread.start()
    
    # Give generation a moment to start
    time.sleep(0.5)
    
    # Consume audio chunks from the streamer
    print("📻 Collecting audio chunks...")
    audio_chunks = []
    chunk_count = 0
    total_samples = 0
    
    # Get stream for first (and only) sample
    audio_stream = audio_streamer.get_stream(0)
    
    for audio_chunk in audio_stream:
        if audio_chunk is None:
            break
            
        # Convert to numpy if needed
        if torch.is_tensor(audio_chunk):
            # Handle bfloat16 -> float32 conversion
            if audio_chunk.dtype == torch.bfloat16:
                audio_chunk = audio_chunk.float()
            audio_np = audio_chunk.cpu().numpy().astype(np.float32)
        else:
            audio_np = audio_chunk
            
        chunk_count += 1
        samples = len(audio_np) if audio_np.ndim == 1 else audio_np.shape[-1]
        total_samples += samples
        
        print(f"  Chunk {chunk_count}: shape={audio_np.shape}, "
              f"dtype={audio_np.dtype}, min={audio_np.min():.3f}, "
              f"max={audio_np.max():.3f}, samples={samples}")
        
        # Ensure 1D
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
            
        audio_chunks.append(audio_np)
    
    # Wait for generation to complete
    generation_thread.join()
    
    # Check for generation errors
    if generation_exception:
        raise generation_exception
    
    if not audio_chunks:
        raise RuntimeError("No audio chunks received from streamer!")
    
    # Concatenate all chunks
    print(f"🎵 Concatenating {chunk_count} chunks, {total_samples} total samples...")
    full_audio = np.concatenate(audio_chunks)
    
    # Ensure proper range for audio
    if full_audio.max() > 1.0 or full_audio.min() < -1.0:
        print(f"  Normalizing audio (range was [{full_audio.min():.3f}, {full_audio.max():.3f}])")
        full_audio = full_audio / np.abs(full_audio).max()
    
    # Save to WAV
    print(f"💾 Writing WAV file: {output_path}")
    sf.write(output_path, full_audio, 24000)
    
    duration = len(full_audio) / 24000
    print(f"✨ Success! Generated {duration:.2f} seconds of audio")
    print(f"   File: {output_path}")
    print(f"   Samples: {len(full_audio)}")
    print(f"   Sample rate: 24000 Hz")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, this is a test of VibeVoice text to speech synthesis."
    
    output_file = synthesize_to_wav(text)
    print(f"\n🎧 Play the audio with: play {output_file}")
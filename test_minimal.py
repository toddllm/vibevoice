#!/usr/bin/env python3
"""
Minimal test of VibeVoice TTS generation
"""

import torch
import numpy as np
import soundfile as sf
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and processor
    print("Loading model...")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-1.5B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    
    processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
    print("Model loaded!")
    
    # Load a voice sample
    voice_path = "demo/voices/en-Alice_woman.wav"
    wav, sr = sf.read(voice_path)
    print(f"Loaded voice: {voice_path}, shape: {wav.shape}, sr: {sr}")
    
    # Prepare text
    text = "Speaker 0: Hello, this is a test."
    print(f"Text: {text}")
    
    # Process inputs
    inputs = processor(
        text=[text],
        voice_samples=[[wav]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Move to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Print input shapes
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"Input {k}: shape {v.shape}")
    
    # Create audio streamer
    audio_streamer = AudioStreamer(
        batch_size=1,
        stop_signal=None,
        timeout=None
    )
    
    # Generate
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            max_new_tokens=100,  # Small for test
            do_sample=False,
            cfg_scale=1.3,
            generation_config={'do_sample': False},
            audio_streamer=audio_streamer,
        )
    
    print("Collecting audio...")
    audio_chunks = []
    for chunk in audio_streamer:
        if chunk is not None:
            print(f"Got chunk shape: {chunk.shape}")
            audio_chunks.append(chunk)
    
    if audio_chunks:
        audio = np.concatenate(audio_chunks)
        print(f"Final audio shape: {audio.shape}")
        sf.write("test_output.wav", audio, 24000)
        print("Saved to test_output.wav")
    else:
        print("No audio chunks received!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test VibeVoice WITHOUT audio streaming
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

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
    
    # Resample to 24kHz if needed
    if sr != 24000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        print(f"Resampled to 24kHz, new shape: {wav.shape}")
    
    # Prepare text
    text = "Speaker 0: Hello world."
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
    
    # Generate WITHOUT audio_streamer
    print("Generating without streaming...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            tokenizer=processor.tokenizer,
            max_new_tokens=200,
            do_sample=False,
            cfg_scale=1.3,
            generation_config={'do_sample': False},
            # NO audio_streamer
        )
    
    print(f"Generated outputs shape: {outputs.shape}")
    print(f"Generated outputs dtype: {outputs.dtype}")
    print(f"Generated outputs min/max: {outputs.min()}, {outputs.max()}")
    
    # Save raw output
    np.save("test_output_raw.npy", outputs.cpu().numpy())
    print("Saved raw output to test_output_raw.npy")

if __name__ == "__main__":
    main()
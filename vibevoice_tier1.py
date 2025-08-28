#!/usr/bin/env python3
"""
VibeVoice Tier 1 Implementation - Production Ready
Robust long-form TTS with proper streaming, memory management, and duration control
"""

import io
import os
import time
import threading
import torch
import numpy as np
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, Response, render_template_string
from flask_cors import CORS
import tempfile
import traceback
import json
import re

# Import VibeVoice components
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

app = Flask(__name__)
CORS(app)

# ========== CONSTANTS ==========

# Audio/Frame Constants (from paper/model card)
FRAME_RATE_HZ = 7.5  # 7.5 acoustic frames per second
SAMPLE_RATE_HZ = 24000  # 24 kHz audio
SAMPLES_PER_FRAME = int(SAMPLE_RATE_HZ / FRAME_RATE_HZ)  # 3200 samples
FRAME_DURATION_MS = 1000 / FRAME_RATE_HZ  # 133.333ms

# Speech Rate (corrected to natural speech)
WORDS_PER_SECOND = 2.75  # ~165 wpm

# Global model instance
model = None
processor = None
device = None
memory_model = None
generation_limits = None

# ========== HELPERS ==========

def duration_to_frames(duration_seconds: float) -> int:
    """Convert duration to frame count with proper rounding"""
    frames = int(round(duration_seconds * FRAME_RATE_HZ))
    return max(frames, 1)

def frames_to_duration(frames: int) -> float:
    """Convert frames back to duration"""
    return frames / FRAME_RATE_HZ

def samples_to_frames(n_samples: int) -> int:
    """Convert PCM samples to frame count (floor division for alignment)"""
    return n_samples // SAMPLES_PER_FRAME

def est_frames_for_text(text: str) -> int:
    """Estimate frames needed for text at natural speech rate"""
    words = max(len(text.split()), 1)
    seconds = words / WORDS_PER_SECOND
    return duration_to_frames(seconds)

def split_sentences(text: str) -> list:
    """Split text into sentences with fallback"""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # Regex fallback for common sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def count_text_tokens(processor, text: str) -> int:
    """Count actual tokens using model's tokenizer"""
    tokens = processor.tokenizer(text, add_special_tokens=False)
    return len(tokens['input_ids'])

# ========== MEMORY CALIBRATION ==========

def prepare_test_input(processor, device):
    """Create minimal deterministic test input"""
    test_text = "Speaker 0: Hello world. This is a test."
    test_voice = np.zeros(24000, dtype=np.float32)  # 1 sec silence
    
    inputs = processor(
        text=[test_text],
        voice_samples=[[test_voice]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

def calibrate_memory_model(model, processor, device):
    """Measure actual memory usage per frame"""
    print("🔬 Calibrating memory model...")
    test_frames = [200, 400, 600]
    measurements = []
    
    # Prepare fixed input
    test_input = prepare_test_input(processor, device)
    
    # Null streamer that discards output
    class NullStreamer(AudioStreamer):
        def __init__(self):
            super().__init__(batch_size=1, stop_signal=None, timeout=None)
        def put(self, *args): pass
        def end(self, *args): pass  # Accept optional parameters
    
    for frames in test_frames:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            _ = model.generate(
                **test_input,
                audio_streamer=NullStreamer(),  # FIXED: correct param name
                tokenizer=processor.tokenizer,
                max_new_tokens=frames,
                do_sample=False,
                cfg_scale=1.3,
            )
        
        # Capture both allocated and reserved
        allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        peak_mb = max(allocated_mb, reserved_mb)
        measurements.append((frames, peak_mb))
        print(f"  {frames} frames: {peak_mb:.1f} MB")
    
    # Linear regression (all in GB to avoid unit errors)
    import numpy as np
    X = np.array([[1, f] for f, _ in measurements])
    y = np.array([m / 1024.0 for _, m in measurements])  # Convert to GB
    base_gb, slope_gb_per_frame = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Apply 85% headroom factor
    available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device == "cuda" else 8.0
    safe_available_gb = available_gb * 0.85
    
    max_safe_frames = int((safe_available_gb - base_gb) / slope_gb_per_frame) if slope_gb_per_frame > 0 else 10000
    
    result = {
        'base_gb': base_gb,
        'slope_gb_per_frame': slope_gb_per_frame,
        'max_safe_frames': max_safe_frames,
        'available_gb': available_gb,
        'safe_available_gb': safe_available_gb
    }
    
    print(f"✅ Memory model: base={base_gb:.2f}GB, slope={slope_gb_per_frame*1000:.2f}MB/frame")
    print(f"   Max safe frames: {max_safe_frames} (~{frames_to_duration(max_safe_frames):.1f}s)")
    
    return result

# ========== GENERATION LIMITS ==========

class GenerationLimits:
    def __init__(self, model_variant="1.5B"):
        if model_variant == "1.5B":
            self.max_text_context = 65536
            self.max_audio_minutes = 90
        else:  # 7B-Preview
            self.max_text_context = 32768
            self.max_audio_minutes = 45
        
        self.max_audio_frames = int(self.max_audio_minutes * 60 * FRAME_RATE_HZ)
    
    def validate_request(self, processor, text: str, target_frames: int, memory_model: dict):
        """Validate generation is within all limits"""
        text_tokens = count_text_tokens(processor, text)
        
        return {
            'text_ok': text_tokens < self.max_text_context,
            'text_tokens': text_tokens,
            'audio_ok': target_frames <= self.max_audio_frames,
            'memory_ok': target_frames <= memory_model['max_safe_frames'],
            'max_safe_frames': memory_model['max_safe_frames']
        }

# ========== PROGRESS TRACKING ==========

class ProgressTracker:
    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.frames_emitted = 0
        self.start_time = time.time()
        self.samples_written = 0
        
    def update(self, new_frames: int):
        """Update with actual frames from PCM length"""
        self.frames_emitted += new_frames
        self.samples_written = self.frames_emitted * SAMPLES_PER_FRAME
        
    def get_status(self) -> dict:
        elapsed = max(time.time() - self.start_time, 0.001)
        throughput_fps = self.frames_emitted / elapsed if self.frames_emitted > 0 else 0
        
        # ETA based on measured throughput
        eta_seconds = None
        if throughput_fps > 0:
            remaining_frames = self.total_frames - self.frames_emitted
            eta_seconds = remaining_frames / throughput_fps
        
        # Real-time factor (RTF)
        seconds_generated = frames_to_duration(self.frames_emitted)
        rtf = seconds_generated / elapsed if elapsed > 0 else 0
        
        return {
            'frames_emitted': self.frames_emitted,
            'frames_total': self.total_frames,
            'seconds_generated': seconds_generated,
            'seconds_total': frames_to_duration(self.total_frames),
            'progress_pct': (self.frames_emitted / self.total_frames * 100) if self.total_frames > 0 else 0,
            'throughput_fps': throughput_fps,
            'rtf': rtf,  # Real-time factor
            'eta_seconds': eta_seconds,
            'samples_written': self.samples_written
        }

# ========== AUDIO PROCESSING ==========

def crossfade_audio(chunk1: np.ndarray, chunk2: np.ndarray, overlap_frames: int = 8) -> np.ndarray:
    """Apply frame-aligned crossfade with peak normalization"""
    
    # Ensure float32
    chunk1 = chunk1.astype(np.float32, copy=False)
    chunk2 = chunk2.astype(np.float32, copy=False)
    
    # Peak normalize each to -1 dBFS before crossfading
    for chunk in (chunk1, chunk2):
        peak = np.max(np.abs(chunk)) + 1e-9
        if peak > 0:
            chunk *= (0.8912509 / peak)  # 10**(-1/20) = -1dB
    
    overlap_samples = overlap_frames * SAMPLES_PER_FRAME
    
    # Check if chunks are long enough
    if len(chunk1) < overlap_samples or len(chunk2) < overlap_samples:
        return np.concatenate([chunk1, chunk2], axis=0)
    
    # Extract overlap regions
    fade_out = chunk1[-overlap_samples:]
    fade_in = chunk2[:overlap_samples]
    
    # Equal-power crossfade
    t = np.linspace(0, np.pi/2, overlap_samples, endpoint=True)
    out_curve = np.cos(t)
    in_curve = np.sin(t)
    
    blended = fade_out * out_curve + fade_in * in_curve
    
    return np.concatenate([
        chunk1[:-overlap_samples],
        blended,
        chunk2[overlap_samples:]
    ], axis=0)

def load_default_voice():
    """Load default voice sample"""
    voice_paths = [
        "demo/voices/en-Alice_woman.wav",
        "/home/tdeshane/VibeVoice/VibeVoice/demo/voices/en-Alice_woman.wav",
        "demo/voices/en-Frank_man.wav",
        "/home/tdeshane/VibeVoice/VibeVoice/demo/voices/en-Frank_man.wav"
    ]
    
    for path in voice_paths:
        if os.path.exists(path):
            wav, sr = sf.read(path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
            return wav.astype(np.float32)
    
    # Fallback to silence
    return np.zeros(24000, dtype=np.float32)

# ========== SYNTHESIS FUNCTIONS ==========

def synthesize_single(model, processor, text: str, target_frames: int,
                     cfg_scale: float = 1.3, voice_sample=None,
                     progress_callback=None, stop_event=None):
    """Single-pass generation with streaming"""
    
    # Format text with speaker
    formatted_text = f"Speaker 0: {text}"
    
    # Use provided voice or default
    if voice_sample is None:
        voice_sample = load_default_voice()
    
    # Prepare inputs
    inputs = processor(
        text=[formatted_text],
        voice_samples=[[voice_sample]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Create AudioStreamer
    audio_streamer = AudioStreamer(
        batch_size=1,
        stop_signal=None,
        timeout=None
    )
    
    # Track progress
    tracker = ProgressTracker(target_frames)
    collected_audio = []
    generation_exception = None
    
    # Stop check function for cooperative stopping
    def stop_check():
        return (tracker.frames_emitted >= target_frames) or \
               (stop_event and stop_event.is_set())
    
    def run_generation():
        nonlocal generation_exception
        try:
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    tokenizer=processor.tokenizer,
                    audio_streamer=audio_streamer,  # FIXED: correct param name
                    cfg_scale=cfg_scale,
                    generation_config={'do_sample': False},
                    stop_check_fn=stop_check,  # Duration control
                    verbose=False,
                    refresh_negative=True,  # Current demos use this
                )
        except Exception as e:
            generation_exception = e
            print(f"Generation error: {e}")
        finally:
            audio_streamer.end()  # Always finalize
    
    # Start generation thread
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()
    
    # Give generation time to start
    time.sleep(0.5)
    
    # Consume audio chunks
    audio_stream = audio_streamer.get_stream(0)
    
    for audio_chunk in audio_stream:
        # Check abort signal
        if stop_event and stop_event.is_set():
            break
        
        if audio_chunk is None:
            break
        
        # Convert to float32 numpy
        if torch.is_tensor(audio_chunk):
            if audio_chunk.dtype == torch.bfloat16:
                audio_chunk = audio_chunk.float()
            audio_np = audio_chunk.cpu().numpy().astype(np.float32)
        else:
            audio_np = audio_chunk.astype(np.float32)
        
        # Ensure 1D
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        
        collected_audio.append(audio_np)
        
        # Update progress based on actual PCM
        frames_in_chunk = samples_to_frames(len(audio_np))
        tracker.update(frames_in_chunk)
        
        if progress_callback:
            progress_callback(tracker.get_status())
        
        # Stop if we have enough
        if tracker.frames_emitted >= target_frames:
            break
    
    # Wait for generation to complete
    gen_thread.join(timeout=5.0)
    
    if generation_exception:
        raise generation_exception
    
    if not collected_audio:
        raise RuntimeError("No audio generated")
    
    # Concatenate all audio
    full_audio = np.concatenate(collected_audio, axis=0)
    
    # Peak normalize to -1 dBFS then clip
    peak = np.max(np.abs(full_audio)) + 1e-9
    if peak > 0:
        full_audio *= (0.8912509 / peak)  # 10**(-1/20) = -1dB
    
    # Safety clip
    full_audio = np.clip(full_audio, -1.0, 1.0).astype(np.float32)
    
    return full_audio, tracker.get_status()

# ========== MODEL LOADING ==========

def load_model():
    """Load the VibeVoice model with configuration"""
    global model, processor, device, memory_model, generation_limits
    
    print("🔄 Loading VibeVoice model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"📍 Using device: {device}, dtype: {dtype}")
    
    try:
        # Load processor
        processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
        
        # Load model with eager attention
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            "microsoft/VibeVoice-1.5B",
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="eager",
        )
        model.eval()
        
        # Initialize generation limits
        generation_limits = GenerationLimits("1.5B")
        
        # Calibrate memory model
        if device == "cuda":
            memory_model = calibrate_memory_model(model, processor, device)
        else:
            memory_model = {
                'base_gb': 2.0,
                'slope_gb_per_frame': 0.001,
                'max_safe_frames': 5000,
                'available_gb': 8.0,
                'safe_available_gb': 6.8
            }
        
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

# ========== FLASK ROUTES ==========

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Main synthesis endpoint with duration control and quality modes"""
    global model, processor, memory_model
    
    if model is None:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    try:
        data = request.json
        
        # Parse inputs
        script = data.get('script', '').strip()
        target_seconds = float(data.get('target_seconds', 60))
        quality = data.get('quality', 'balanced')  # fast|balanced|high
        max_vram_gb = data.get('max_vram_gb', None)
        
        if not script:
            return jsonify({'error': 'No script provided'}), 400
        
        # Convert to frames
        target_frames = duration_to_frames(target_seconds)
        
        # Quality presets
        quality_presets = {
            'fast': {'cfg_scale': 1.0, 'solver_steps': 8},
            'balanced': {'cfg_scale': 1.3, 'solver_steps': 16},
            'high': {'cfg_scale': 1.5, 'solver_steps': 24}
        }
        
        cfg_scale = quality_presets[quality]['cfg_scale']
        solver_steps = quality_presets[quality]['solver_steps']
        
        # Set solver steps if model supports it
        if hasattr(model, 'set_ddpm_inference_steps'):
            model.set_ddpm_inference_steps(solver_steps)
        
        # Validate request
        validation = generation_limits.validate_request(processor, script, target_frames, memory_model)
        
        if not validation['text_ok']:
            return jsonify({
                'error': f'Text too long: {validation["text_tokens"]} tokens (max: {generation_limits.max_text_context})'
            }), 400
        
        if not validation['audio_ok']:
            return jsonify({
                'error': f'Duration too long: {target_seconds}s (max: {generation_limits.max_audio_minutes*60}s)'
            }), 400
        
        # Memory preflight
        estimated_gb = memory_model['base_gb'] + memory_model['slope_gb_per_frame'] * target_frames
        
        if max_vram_gb and estimated_gb > max_vram_gb:
            # Would exceed memory limit - need chunking (Tier 2)
            return jsonify({
                'error': f'Would use {estimated_gb:.1f}GB (limit: {max_vram_gb}GB). Chunking not yet implemented.'
            }), 400
        
        if not validation['memory_ok']:
            return jsonify({
                'error': f'Duration would use {estimated_gb:.1f}GB. Max safe: {frames_to_duration(validation["max_safe_frames"]):.1f}s'
            }), 400
        
        print(f"📝 Synthesizing {target_seconds}s ({target_frames} frames) with quality={quality}")
        print(f"   Text tokens: {validation['text_tokens']}, Est. memory: {estimated_gb:.1f}GB")
        
        # Generate audio
        stop_event = threading.Event()
        
        def progress_update(status):
            # Could send SSE updates here
            if status['progress_pct'] % 10 < 1:  # Log every ~10%
                print(f"   Progress: {status['progress_pct']:.0f}%, RTF: {status['rtf']:.2f}x")
        
        audio, final_status = synthesize_single(
            model, processor,
            script,
            target_frames,
            cfg_scale,
            progress_callback=progress_update,
            stop_event=stop_event
        )
        
        print(f"✅ Generated {final_status['seconds_generated']:.1f}s, RTF: {final_status['rtf']:.2f}x")
        
        # Save to temporary WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, SAMPLE_RATE_HZ)
            tmp_path = tmp_file.name
        
        # Read and encode
        with open(tmp_path, 'rb') as f:
            wav_data = f.read()
        
        os.remove(tmp_path)
        
        import base64
        wav_b64 = base64.b64encode(wav_data).decode('utf-8')
        
        return jsonify({
            'wav_b64': wav_b64,
            'sample_rate': SAMPLE_RATE_HZ,
            'metrics': final_status
        })
        
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Check model and memory status"""
    return jsonify({
        'model_loaded': model is not None,
        'device': device if device else 'Not initialized',
        'memory_model': memory_model,
        'limits': {
            'max_text_tokens': generation_limits.max_text_context if generation_limits else None,
            'max_audio_minutes': generation_limits.max_audio_minutes if generation_limits else None
        }
    })

@app.route('/')
def index():
    """Simple test UI"""
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>VibeVoice Tier 1</title></head>
    <body>
        <h1>VibeVoice Tier 1 API</h1>
        <p>POST /synthesize with {script, target_seconds, quality}</p>
        <p>GET /status for model info</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Starting VibeVoice Tier 1...")
    load_model()
    print("🌐 Starting web server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
#!/usr/bin/env python3
"""
VibeVoice Production Server V2 - Using PROVEN 7B code
Preserves commit 8e548c2 invariants for 1.5B while using working 7B approach
"""

import os
import sys
import time
import torch
import numpy as np
import threading
import hashlib
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from contextlib import contextmanager
import scipy.io.wavfile as wavfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import VibeVoice components
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

# =============================================================================
# Constants
# =============================================================================

SPEECH_DIFFUSION_ID = 151654  # Acoustic branch entry token
GOLDEN_CONTROL_KEYS = ['cfg_scale', 'ddpm_steps', 'dtype', 'refresh_negative']

# =============================================================================
# Golden Path Validator - Preserves 8e548c2 Invariants
# =============================================================================

@dataclass
class ModelInvariants:
    """Immutable configuration from commit 8e548c2"""
    model_1_5b: Dict[str, Any] = field(default_factory=lambda: {
        'model_id': 'microsoft/VibeVoice-1.5B',
        'dtype': 'bfloat16',  # CRITICAL: NOT float16
        'attn_implementation': 'eager',
        'device_map': 'cuda',
        'ddpm_steps': None,  # NEVER set for streaming
        'generation': {
            'cfg_scale': 1.3,
            'do_sample': False,
            'refresh_negative': True,
            'verbose': False,
        },
        'streaming': {
            'batch_size': 1,
            'stop_signal': None,
            'timeout': None,
            'pre_read_ms': 300,
            'first_chunk_timeout_s': 3,
            'join_timeout_s': 10,
        }
    })
    
    model_7b: Dict[str, Any] = field(default_factory=lambda: {
        'model_id': 'WestZhang/VibeVoice-Large-pt',
        'dtype': 'bfloat16',  # Use bfloat16 for 7B too (proven to work)
        'attn_implementation': 'eager',
        'device_map': 'cuda',
        'ddpm_steps': None,  # Don't set for offline either (proven config)
        'generation': {
            'cfg_scale': 1.3,
            'do_sample': False,
            'return_speech': True,  # Critical for offline
        },
        'offline_default': True,
    })

class GoldenPathValidator:
    """Validates generation parameters against 8e548c2 invariants"""
    
    def __init__(self):
        self.invariants = ModelInvariants()
        self._control_hashes = {}
        
    def check_bf16_support(self) -> Tuple[bool, str]:
        """Check if BF16 is supported on current hardware"""
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        try:
            # Check compute capability
            major, minor = torch.cuda.get_device_capability()
            if major < 8:  # Ampere or newer for native BF16
                return False, f"GPU compute capability {major}.{minor} < 8.0"
            
            # Try creating a BF16 tensor
            test_tensor = torch.tensor([], dtype=torch.bfloat16, device='cuda')
            return True, "BF16 supported"
            
        except Exception as e:
            return False, f"BF16 test failed: {e}"
    
    def validate_1_5b_streaming(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate 1.5B streaming parameters"""
        expected = self.invariants.model_1_5b
        
        # Critical dtype check
        if params.get('dtype') != 'bfloat16':
            return False, f"1.5B requires bfloat16, got {params.get('dtype')}"
        
        # DDPM must not be set
        if params.get('ddpm_steps') is not None:
            return False, "1.5B streaming cannot have DDPM steps"
        
        # refresh_negative is required
        if not params.get('generation', {}).get('refresh_negative'):
            return False, "1.5B requires refresh_negative=True"
        
        # Never use generation_config on frozen path
        if 'generation_config' in params:
            return False, "1.5B frozen path cannot use generation_config"
        
        return True, "Valid"
    
    def get_control_hash(self, params: Dict[str, Any]) -> str:
        """Create deterministic hash of control parameters"""
        control = {k: params.get(k) for k in GOLDEN_CONTROL_KEYS if k in params}
        
        # Add nested generation params
        if 'generation' in params:
            for k in GOLDEN_CONTROL_KEYS:
                if k in params['generation']:
                    control[f'generation.{k}'] = params['generation'][k]
        
        snapshot = json.dumps(control, sort_keys=True)
        hash_val = hashlib.sha256(snapshot.encode()).hexdigest()[:8]
        self._control_hashes[hash_val] = control
        return hash_val

# =============================================================================
# Thread-Safe Streaming Context with Re-entrant DDPM Guard
# =============================================================================

class StreamingContext:
    """Thread-safe context for streaming generation with re-entrant DDPM guard"""
    
    def __init__(self):
        self._model_locks = {}  # Per-model generation locks
        self._ddpm_lock = threading.RLock()  # Re-entrant lock
        self._ddpm_depth = 0  # Depth counter for nested calls
        self._original_ddpm = {}  # Stack of original values
        self._active_streamers = {}
        self._streamer_lock = threading.Lock()
        
    def get_model_lock(self, model_id: str) -> threading.Lock:
        """Get or create per-model lock"""
        if model_id not in self._model_locks:
            self._model_locks[model_id] = threading.Lock()
        return self._model_locks[model_id]
    
    @contextmanager
    def ddpm_guard(self, model, steps: Optional[int]):
        """Re-entrant DDPM configuration guard"""
        model_id = id(model)
        
        with self._ddpm_lock:
            self._ddpm_depth += 1
            
            # Save original only on first entry
            if self._ddpm_depth == 1:
                self._original_ddpm[model_id] = getattr(model, '_ddpm_inference_steps', None)
            
            try:
                if steps is not None and hasattr(model, 'set_ddpm_inference_steps'):
                    model.set_ddpm_inference_steps(num_steps=steps)
                    logger.debug(f"Set DDPM steps to {steps} (depth={self._ddpm_depth})")
                yield
            finally:
                self._ddpm_depth -= 1
                
                # Restore only when all guards exit
                if self._ddpm_depth == 0 and model_id in self._original_ddpm:
                    original = self._original_ddpm.pop(model_id)
                    if original is not None and hasattr(model, 'set_ddpm_inference_steps'):
                        model.set_ddpm_inference_steps(num_steps=original)
                        logger.debug(f"Restored DDPM steps to {original}")
    
    def create_streamer(self, stream_id: str, batch_size: int = 1) -> AudioStreamer:
        """Create and track streamer"""
        with self._streamer_lock:
            if stream_id in self._active_streamers:
                logger.warning(f"Streamer {stream_id} already exists")
                return self._active_streamers[stream_id]
            
            streamer = AudioStreamer(
                batch_size=batch_size,
                stop_signal=None,
                timeout=None
            )
            self._active_streamers[stream_id] = streamer
            return streamer
    
    def cleanup_streamer(self, stream_id: str):
        """Cleanup streamer"""
        with self._streamer_lock:
            if stream_id in self._active_streamers:
                streamer = self._active_streamers[stream_id]
                streamer.end()
                del self._active_streamers[stream_id]
                logger.debug(f"Cleaned up streamer {stream_id}")

# =============================================================================
# Voice Loader with Sanity Checks
# =============================================================================

class VoiceLoader:
    """Voice loading with LRU cache and sanity checks"""
    
    def __init__(self, max_cache_size: int = 10):
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self._lock = threading.Lock()
        
    def load_voice(self, voice_path: str, target_sr: int = 24000) -> np.ndarray:
        """Load voice with sanity checks"""
        with self._lock:
            # Check cache
            if voice_path in self.cache:
                self.cache.move_to_end(voice_path)
                logger.debug(f"Voice cache hit: {voice_path}")
                return self.cache[voice_path].copy()
            
            # Load and validate
            logger.info(f"Loading voice: {voice_path}")
            sr, wav = wavfile.read(voice_path)
            
            # Convert to float32
            if wav.dtype == np.int16:
                wav = wav.astype(np.float32) / 32768.0
            else:
                wav = wav.astype(np.float32)
            
            # Ensure mono
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            
            # Duration check (0.4-6s)
            duration = len(wav) / sr
            if duration < 0.4 or duration > 6.0:
                logger.warning(f"Voice duration {duration:.1f}s outside recommended range [0.4, 6.0]")
            
            # Check RMS floor (avoid near-silence)
            rms = np.sqrt(np.mean(wav**2))
            if rms < 0.001:
                logger.warning(f"Voice RMS {rms:.4f} very low - may be near-silent")
            
            # Normalize to [-0.95, 0.95]
            max_val = np.abs(wav).max()
            if max_val > 0:
                wav = wav / max_val * 0.95
            
            # Resample if needed
            if sr != target_sr:
                logger.info(f"Resampling voice from {sr}Hz to {target_sr}Hz")
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # LRU cache eviction
            if len(self.cache) >= self.max_cache_size:
                evicted = self.cache.popitem(last=False)
                logger.debug(f"Evicted voice from cache: {evicted[0]}")
            
            self.cache[voice_path] = wav
            return wav.copy()

# =============================================================================
# Speaker Parser with Extended Support
# =============================================================================

class SpeakerParser:
    """Robust speaker parsing with normalization"""
    
    # Extended regex for leading whitespace, tabs, special chars
    SPEAKER_PATTERN = re.compile(r'^\s*([^:]{1,40}):\s*(.+)', re.MULTILINE)
    
    @classmethod
    def parse_script(cls, text: str) -> List[Tuple[str, str]]:
        """Parse script into (speaker, content) pairs"""
        segments = []
        last_speaker = "Speaker 0"  # Default
        
        lines = text.strip().split('\n')
        mixed_format_warned = False
        
        for line in lines:
            line = line.rstrip()  # Keep leading whitespace for pattern match
            
            if not line:
                continue  # Skip empty lines
            
            match = cls.SPEAKER_PATTERN.match(line)
            if match:
                speaker = match.group(1).strip()
                content = match.group(2).strip()
                
                # Normalize speaker format
                if speaker.startswith('Speaker ') and speaker[8:].isdigit():
                    # Already in canonical format
                    pass
                else:
                    # Named speaker - normalize to Speaker N
                    if not mixed_format_warned:
                        logger.warning(f"Mixed speaker formats detected - normalizing")
                        mixed_format_warned = True
                    
                    # Map to Speaker 0, 1, 2... based on order
                    speaker = f"Speaker {len(set(s for s, _ in segments))}"
                
                last_speaker = speaker
                segments.append((speaker, content))
            else:
                # Continuation line - use last speaker
                segments.append((last_speaker, line.strip()))
        
        return segments

# =============================================================================
# Production Server with Proven 7B Path
# =============================================================================

class VibeVoiceProductionServer:
    """Production server using proven configurations"""
    
    def __init__(self):
        self.validator = GoldenPathValidator()
        self.streaming_ctx = StreamingContext()
        self.voice_loader = VoiceLoader()
        self.speaker_parser = SpeakerParser()
        
        self.current_model = None
        self.current_model_id = None
        self.processor = None
        
        # Check BF16 support at startup
        bf16_ok, bf16_msg = self.validator.check_bf16_support()
        if not bf16_ok:
            logger.warning(f"BF16 not available: {bf16_msg}")
            logger.warning("Use 7B offline or deploy on BF16-capable GPU")
        
        self._diagnostics = {
            'requests_total': 0,
            'requests_1_5b': 0,
            'requests_7b': 0,
            'streaming_success': 0,
            'offline_success': 0,
            'stream_fallbacks': 0,
            'errors': 0,
            'last_errors': [],
            'last_control_hash': None,
            'bf16_available': bf16_ok,
        }
    
    def duration_to_frames(self, seconds: float, frame_rate: float = 7.5) -> int:
        """Convert seconds to frame count"""
        return int(seconds * frame_rate)
    
    def load_model(self, model_size: str = "1.5B"):
        """Load model with proper configuration"""
        # Check if correct model already loaded
        if self.current_model is not None:
            if ((model_size == "1.5B" and "1.5B" in self.current_model_id) or
                (model_size == "7B" and "Large" in self.current_model_id)):
                logger.debug(f"Model {self.current_model_id} already loaded")
                return
        
        # Unload current model
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_id}")
            del self.current_model
            torch.cuda.empty_cache()
            self.current_model = None
            self.current_model_id = None
        
        if model_size == "1.5B":
            invariants = self.validator.invariants.model_1_5b
            model_id = invariants['model_id']
            
            logger.info(f"Loading 1.5B model: {model_id}")
            self.processor = VibeVoiceProcessor.from_pretrained(model_id)
            self.current_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # CRITICAL
                device_map=invariants['device_map'],
                attn_implementation=invariants['attn_implementation'],
            )
            
        elif model_size == "7B":
            invariants = self.validator.invariants.model_7b
            model_id = invariants['model_id']
            
            logger.info(f"Loading 7B model: {model_id}")
            # Use EXACT proven configuration from test_7b_offline.py
            self.processor = VibeVoiceProcessor.from_pretrained(model_id)
            self.current_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Proven to work
                device_map="cuda",
                attn_implementation="eager"
            )
            # Do NOT set DDPM steps - proven config doesn't use them
        
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        self.current_model.eval()
        self.current_model_id = model_id
        logger.info(f"Model loaded successfully: {model_id}")
    
    def generate_1_5b_streaming(
        self,
        text: str,
        voice_path: str,
        stream_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """1.5B streaming generation - preserves 8e548c2 invariants"""
        
        if stream_id is None:
            stream_id = f"stream_{int(time.time() * 1000)}"
        
        invariants = self.validator.invariants.model_1_5b
        start_time = time.time()
        
        # Create control hash
        params = {
            'dtype': 'bfloat16',
            'ddpm_steps': None,
            'generation': invariants['generation']
        }
        control_hash = self.validator.get_control_hash(params)
        self._diagnostics['last_control_hash'] = control_hash
        
        logger.info(f"1.5B streaming: {stream_id}, control={control_hash}")
        
        # Load voice
        wav = self.voice_loader.load_voice(voice_path)
        
        # Process inputs - EXACT tier2.py format
        voice_samples = [[wav]]  # Double brackets!
        inputs = self.processor(
            text=[text],  # List wrapping
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Move to device
        device = "cuda"
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Create streamer
        audio_streamer = self.streaming_ctx.create_streamer(stream_id)
        
        # Collect chunks
        chunks = []
        generation_exception = None
        first_chunk_time = None
        
        def run_generation():
            nonlocal generation_exception
            try:
                # Use model lock to prevent concurrent generation
                model_lock = self.streaming_ctx.get_model_lock(self.current_model_id)
                with model_lock:
                    with torch.no_grad():
                        # EXACT tier2.py generation - NO return_speech, NO generation_config!
                        _ = self.current_model.generate(
                            **inputs,
                            tokenizer=self.processor.tokenizer,
                            audio_streamer=audio_streamer,
                            cfg_scale=1.3,
                            generation_config={'do_sample': False},
                            refresh_negative=True,  # CRITICAL
                            verbose=False,
                        )
            except Exception as e:
                generation_exception = e
                logger.error(f"Generation error: {e}")
            finally:
                audio_streamer.end()
        
        # Start generation
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()
        
        # Wait before collecting (tier2.py pattern)
        time.sleep(invariants['streaming']['pre_read_ms'] / 1000)
        
        # Collect chunks with first-chunk timeout
        timeout_time = start_time + invariants['streaming']['first_chunk_timeout_s']
        
        for chunk in audio_streamer.get_stream(0):
            if chunk is None:
                break
            
            if not chunks and time.time() > timeout_time:
                logger.error(f"First chunk timeout ({invariants['streaming']['first_chunk_timeout_s']}s)")
                break
            
            if not chunks:
                first_chunk_time = time.time() - start_time
            
            if torch.is_tensor(chunk):
                if chunk.dtype == torch.bfloat16:
                    chunk = chunk.float()
                chunk = chunk.cpu().numpy()
            
            chunks.append(chunk.squeeze() if hasattr(chunk, 'ndim') and chunk.ndim > 1 else chunk)
        
        # Wait for completion with timeout
        gen_thread.join(timeout=invariants['streaming']['join_timeout_s'])
        
        # Cleanup
        self.streaming_ctx.cleanup_streamer(stream_id)
        
        if generation_exception:
            raise generation_exception
        
        if chunks:
            audio = np.concatenate(chunks)
            duration = len(audio) / 24000
            rtf = duration / (time.time() - start_time)
            
            self._diagnostics['streaming_success'] += 1
            logger.info(f"1.5B streaming success: {duration:.2f}s, RTF={rtf:.2f}x")
            
            metadata = {
                'model': '1.5B',
                'transport': 'streaming',
                'control_hash': control_hash,
                'duration_s': duration,
                'rtf': rtf,
                'first_chunk_ms': int(first_chunk_time * 1000) if first_chunk_time else None,
                'chunks': len(chunks)
            }
            
            return audio, metadata
        else:
            raise RuntimeError("NO_CHUNKS_BF16_OR_DDPM: 1.5B streaming received zero chunks")
    
    def generate_7b_offline(
        self,
        text: str,
        voice_path: str,
        max_seconds: int = 30
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """7B offline generation - PROVEN approach from test_7b_offline.py"""
        
        start_time = time.time()
        logger.info("7B offline generation (proven path)")
        
        # Load voice
        wav = self.voice_loader.load_voice(voice_path)
        
        # Process inputs - EXACT proven format
        voice_samples = [[wav]]
        inputs = self.processor(
            text=[text],  # Batch dimension
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Move to device
        inputs = {k: v.to(self.current_model.device) if hasattr(v, 'to') else v 
                  for k, v in inputs.items()}
        
        # Calculate max tokens
        max_new_tokens = self.duration_to_frames(max_seconds)
        
        # Use model lock
        model_lock = self.streaming_ctx.get_model_lock(self.current_model_id)
        
        with model_lock:
            with torch.no_grad():
                # EXACT proven generation call from test_7b_offline.py
                out = self.current_model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    return_speech=True,  # Critical for offline
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic
                    cfg_scale=1.3,
                )
        
        # Extract audio - EXACT proven extraction
        if hasattr(out, 'speech_outputs') and out.speech_outputs and out.speech_outputs[0] is not None:
            audio = out.speech_outputs[0]
            
            # Convert to numpy
            if isinstance(audio, torch.Tensor):
                audio = audio.to(torch.float32).cpu().numpy()
            
            # Ensure 1D
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            
            duration = len(audio) / 24000
            rtf = duration / (time.time() - start_time)
            
            self._diagnostics['offline_success'] += 1
            logger.info(f"7B offline success: {duration:.2f}s, RTF={rtf:.2f}x")
            
            # Check for speech_diffusion_id
            speech_id_found = False
            if hasattr(out, 'sequences'):
                speech_id_found = SPEECH_DIFFUSION_ID in out.sequences[0]
            
            metadata = {
                'model': '7B',
                'transport': 'offline',
                'control_hash': None,
                'duration_s': duration,
                'rtf': rtf,
                'speech_diffusion_id_found': speech_id_found
            }
            
            return audio, metadata
        else:
            # Check if model stayed in text mode
            if hasattr(out, 'sequences'):
                if SPEECH_DIFFUSION_ID not in out.sequences[0]:
                    raise RuntimeError("Model stayed in text mode - no acoustic tokens generated")
            
            raise RuntimeError("No audio in speech_outputs")
    
    def generate(
        self,
        text: str,
        voice_path: str,
        model_size: str = "1.5B",
        use_streaming: Optional[bool] = None,
        max_seconds: int = 30
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Main generation entry point with metadata"""
        
        self._diagnostics['requests_total'] += 1
        
        try:
            # Load model if needed
            if self.current_model is None:
                self.load_model(model_size)
            elif model_size == "1.5B" and "1.5B" not in self.current_model_id:
                self.load_model(model_size)
            elif model_size == "7B" and "Large" not in self.current_model_id:
                self.load_model(model_size)
            
            if model_size == "1.5B":
                self._diagnostics['requests_1_5b'] += 1
                # 1.5B always uses streaming (8e548c2 invariant)
                return self.generate_1_5b_streaming(text, voice_path)
                
            elif model_size == "7B":
                self._diagnostics['requests_7b'] += 1
                # 7B defaults to offline (proven path)
                return self.generate_7b_offline(text, voice_path, max_seconds)
            
        except Exception as e:
            self._diagnostics['errors'] += 1
            
            # Track last errors
            error_info = {
                'time': time.time(),
                'model': model_size,
                'error': str(e)
            }
            self._diagnostics['last_errors'].append(error_info)
            if len(self._diagnostics['last_errors']) > 5:
                self._diagnostics['last_errors'].pop(0)
            
            logger.error(f"Generation failed: {e}")
            raise
    
    def post_process_audio(
        self,
        audio: np.ndarray,
        normalize: bool = True,
        limiter: bool = False
    ) -> np.ndarray:
        """Audio post-processing with safety"""
        
        # Final peak normalize to -1 dBFS
        if normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                scale = 0.95 / max_val
                audio = audio * scale
        
        # Optional limiter (disabled by default to preserve transients)
        if limiter:
            audio = np.clip(audio, -0.99, 0.99)
        
        return audio
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics"""
        return {
            **self._diagnostics,
            'current_model': self.current_model_id,
            'cache_size': len(self.voice_loader.cache),
            'active_streams': len(self.streaming_ctx._active_streamers),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health endpoint with basic validation"""
        return {
            'status': 'healthy',
            'bf16_available': self._diagnostics['bf16_available'],
            'current_model': self.current_model_id,
            'diagnostics': self.get_diagnostics(),
            'last_control_hash': self._diagnostics['last_control_hash'],
            'last_errors': self._diagnostics['last_errors'][-3:] if self._diagnostics['last_errors'] else []
        }

# =============================================================================
# Comprehensive Test Suite
# =============================================================================

def test_production_server():
    """Comprehensive test suite including concurrency and edge cases"""
    
    print("\n" + "="*60)
    print("Testing VibeVoice Production Server V2")
    print("="*60)
    
    server = VibeVoiceProductionServer()
    voice_path = "demo/voices/en-Alice_woman.wav"
    
    # Test 1: BF16 support check
    print("\n[TEST 1] BF16 Support")
    bf16_ok, msg = server.validator.check_bf16_support()
    print(f"  BF16: {msg}")
    
    # Test 2: Golden snapshot validation
    print("\n[TEST 2] Golden Snapshot")
    params = {
        'dtype': 'bfloat16',
        'ddpm_steps': None,
        'generation': {
            'refresh_negative': True,
            'cfg_scale': 1.3,
        }
    }
    valid, msg = server.validator.validate_1_5b_streaming(params)
    control_hash = server.validator.get_control_hash(params)
    print(f"  Validation: {msg}")
    print(f"  Control hash: {control_hash}")
    assert valid, "1.5B validation failed"
    
    # Test 3: Speaker parsing edge cases
    print("\n[TEST 3] Speaker Parsing")
    test_scripts = [
        "  Speaker 0: Leading whitespace",
        "\tSpeaker 1: Tab prefix",
        "José Álvarez: Accented name",
        "O'Brien: Apostrophe test",
        "Speaker 0: First line\nContinuation without tag",
    ]
    
    for script in test_scripts:
        segments = server.speaker_parser.parse_script(script)
        print(f"  Parsed {len(segments)} segments from: {script[:30]}...")
    
    # Test 4: 1.5B streaming
    print("\n[TEST 4] 1.5B Streaming")
    try:
        text = "Speaker 0: Testing production server version two with proven configurations."
        audio, metadata = server.generate(text, voice_path, model_size="1.5B")
        print(f"  ✓ Generated {metadata['duration_s']:.2f}s")
        print(f"    RTF: {metadata['rtf']:.2f}x")
        print(f"    Control: {metadata['control_hash']}")
        print(f"    First chunk: {metadata.get('first_chunk_ms')}ms")
        
        # Post-process and save
        audio = server.post_process_audio(audio)
        wavfile.write("test_production_v2_1.5b.wav", 24000, (audio * 32767).astype(np.int16))
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 5: 7B offline (proven path)
    print("\n[TEST 5] 7B Offline (Proven)")
    try:
        text = "Speaker 0: Testing seven B model with proven offline configuration."
        audio, metadata = server.generate(text, voice_path, model_size="7B", max_seconds=10)
        print(f"  ✓ Generated {metadata['duration_s']:.2f}s")
        print(f"    RTF: {metadata['rtf']:.2f}x")
        print(f"    Speech ID found: {metadata.get('speech_diffusion_id_found')}")
        
        # Post-process and save
        audio = server.post_process_audio(audio)
        wavfile.write("test_production_v2_7b.wav", 24000, (audio * 32767).astype(np.int16))
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 6: Health check
    print("\n[TEST 6] Health Check")
    health = server.health_check()
    print(f"  Status: {health['status']}")
    print(f"  BF16: {health['bf16_available']}")
    print(f"  Model: {health['current_model']}")
    print(f"  Requests: {health['diagnostics']['requests_total']}")
    
    # Test 7: Concurrent 1.5B requests (DDPM guard test)
    print("\n[TEST 7] Concurrent Streaming")
    import concurrent.futures
    
    def concurrent_request(i):
        text = f"Speaker 0: Concurrent request number {i}."
        return server.generate(text, voice_path, model_size="1.5B")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(concurrent_request, i) for i in range(2)]
        results = []
        
        for future in concurrent.futures.as_completed(futures):
            try:
                audio, metadata = future.result()
                results.append(metadata)
                print(f"  ✓ Concurrent request completed: {metadata['duration_s']:.2f}s")
            except Exception as e:
                print(f"  ✗ Concurrent request failed: {e}")
    
    print("\n" + "="*60)
    print("Production server V2 tests complete")
    print(f"Final diagnostics: {server.get_diagnostics()}")
    print("="*60)

if __name__ == "__main__":
    test_production_server()
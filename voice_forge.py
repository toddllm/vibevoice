#!/usr/bin/env python3
"""
Voice Forge - Scalable voice creation and variant generation
Safe, quality-gated voice import and DSP-based variant creation
"""

import os
import yaml
import hashlib
import numpy as np
import scipy.io.wavfile as wavfile
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QualityMetrics:
    """Voice quality metrics"""
    duration: float
    sample_rate: int
    snr_db: float
    rms_db: float
    clipping_ratio: float
    vad_speech_ratio: float
    passed: bool
    issues: List[str]

@dataclass
class VoiceSegment:
    """Curated voice segment"""
    start: float
    end: float
    duration: float
    snr_db: float
    rms_db: float

# =============================================================================
# Quality Checker
# =============================================================================

class VoiceQualityChecker:
    """Check voice quality with practical gates"""
    
    # Quality thresholds
    MIN_DURATION = 15.0  # Reject < 15s
    WARN_DURATION = 30.0  # Warn < 30s
    MAX_DURATION = 300.0  # 5 minutes max
    MIN_SNR_DB = 20.0
    MIN_RMS_DB = -38.0
    MAX_CLIPPING_RATIO = 0.001  # 0.1%
    MIN_SPEECH_RATIO = 0.5  # 50% must be speech
    
    def analyze(self, audio: np.ndarray, sr: int) -> QualityMetrics:
        """Comprehensive quality analysis"""
        
        issues = []
        duration = len(audio) / sr
        
        # Duration check
        if duration < self.MIN_DURATION:
            issues.append(f"Duration {duration:.1f}s < minimum {self.MIN_DURATION}s")
        elif duration < self.WARN_DURATION:
            issues.append(f"Warning: Duration {duration:.1f}s < recommended {self.WARN_DURATION}s")
        elif duration > self.MAX_DURATION:
            issues.append(f"Duration {duration:.1f}s > maximum {self.MAX_DURATION}s")
        
        # RMS level
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        if rms_db < self.MIN_RMS_DB:
            issues.append(f"RMS {rms_db:.1f}dB too low (< {self.MIN_RMS_DB}dB)")
        
        # Clipping detection
        clipping_samples = np.sum(np.abs(audio) > 0.99)
        clipping_ratio = clipping_samples / len(audio)
        if clipping_ratio > self.MAX_CLIPPING_RATIO:
            issues.append(f"Clipping ratio {clipping_ratio:.4f} > maximum {self.MAX_CLIPPING_RATIO}")
        
        # SNR estimation (simple spectral flatness method)
        snr_db = self._estimate_snr(audio, sr)
        if snr_db < self.MIN_SNR_DB:
            issues.append(f"SNR {snr_db:.1f}dB < minimum {self.MIN_SNR_DB}dB")
        
        # VAD speech ratio
        speech_ratio = self._estimate_speech_ratio(audio, sr)
        if speech_ratio < self.MIN_SPEECH_RATIO:
            issues.append(f"Speech ratio {speech_ratio:.2f} < minimum {self.MIN_SPEECH_RATIO}")
        
        # Overall pass/fail
        critical_fail = (
            duration < self.MIN_DURATION or
            clipping_ratio > self.MAX_CLIPPING_RATIO or
            rms_db < self.MIN_RMS_DB
        )
        
        return QualityMetrics(
            duration=duration,
            sample_rate=sr,
            snr_db=snr_db,
            rms_db=rms_db,
            clipping_ratio=clipping_ratio,
            vad_speech_ratio=speech_ratio,
            passed=not critical_fail,
            issues=issues
        )
    
    def _estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        """Estimate SNR using spectral flatness"""
        # Simple SNR estimation
        # High spectral flatness = more noise-like
        # Low spectral flatness = more tonal/speech-like
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(S=magnitude)
        mean_flatness = np.mean(flatness)
        
        # Convert to approximate SNR (empirical mapping)
        # Lower flatness = higher SNR
        estimated_snr = 40 * (1 - mean_flatness)
        
        return float(estimated_snr)
    
    def _estimate_speech_ratio(self, audio: np.ndarray, sr: int) -> float:
        """Estimate ratio of speech to total duration"""
        # Simple energy-based VAD
        
        # Frame the signal
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)  # 10ms hop
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        
        # Compute energy per frame
        energy = np.sum(frames**2, axis=0)
        
        # Threshold at 20% of max energy
        threshold = 0.2 * np.max(energy)
        speech_frames = energy > threshold
        
        return np.mean(speech_frames)

# =============================================================================
# Voice Curation
# =============================================================================

class VoiceCurator:
    """Select best segments from voice recording"""
    
    def find_best_segments(
        self, 
        audio: np.ndarray, 
        sr: int,
        min_segment: float = 2.0,
        max_segment: float = 6.0,
        num_segments: int = 5
    ) -> List[VoiceSegment]:
        """Find best segments for voice cloning"""
        
        segments = []
        
        # Simple approach: sliding window with quality scoring
        window_size = int(min_segment * sr)
        hop_size = int(0.5 * sr)  # 0.5s hop
        
        for start_idx in range(0, len(audio) - window_size, hop_size):
            # Try different segment lengths
            for seg_duration in [2.0, 3.0, 4.0, 5.0, 6.0]:
                if seg_duration > max_segment:
                    break
                    
                end_idx = start_idx + int(seg_duration * sr)
                if end_idx > len(audio):
                    break
                
                segment = audio[start_idx:end_idx]
                
                # Score segment
                rms = np.sqrt(np.mean(segment**2))
                rms_db = 20 * np.log10(rms + 1e-10)
                
                # Skip if too quiet
                if rms_db < -35:
                    continue
                
                # Check for clipping
                if np.sum(np.abs(segment) > 0.99) > 10:
                    continue
                
                # Simple SNR estimate
                snr = self._quick_snr(segment)
                
                segments.append(VoiceSegment(
                    start=start_idx / sr,
                    end=end_idx / sr,
                    duration=seg_duration,
                    snr_db=snr,
                    rms_db=rms_db
                ))
        
        # Sort by quality (SNR * RMS weight)
        segments.sort(key=lambda s: s.snr_db - 0.1 * abs(s.rms_db + 20), reverse=True)
        
        # Return top segments
        return segments[:num_segments]
    
    def _quick_snr(self, segment: np.ndarray) -> float:
        """Quick SNR estimation for segment"""
        # Estimate noise from quietest 10%
        sorted_abs = np.sort(np.abs(segment))
        noise_floor = np.mean(sorted_abs[:len(sorted_abs)//10])
        signal_peak = np.mean(sorted_abs[-len(sorted_abs)//10:])
        
        if noise_floor > 0:
            snr = 20 * np.log10(signal_peak / noise_floor)
        else:
            snr = 40.0
        
        return float(np.clip(snr, 0, 60))

# =============================================================================
# Variant Generator
# =============================================================================

class VariantGenerator:
    """Generate tasteful voice variants via DSP"""
    
    # Safe variant recipes
    RECIPES = {
        'bright_fast': {
            'name': 'Bright & Fast',
            'ops': [
                {'type': 'rate', 'factor': 1.04},
                {'type': 'pitch', 'semitones': 1},
                {'type': 'eq_shelf', 'freq': 5000, 'gain_db': 2}
            ]
        },
        'warm_slow': {
            'name': 'Warm & Slow',
            'ops': [
                {'type': 'rate', 'factor': 0.96},
                {'type': 'pitch', 'semitones': -1},
                {'type': 'eq_shelf', 'freq': 200, 'gain_db': 1.5}
            ]
        },
        'neutral_mid': {
            'name': 'Neutral Mid',
            'ops': [
                {'type': 'rate', 'factor': 1.0},
                {'type': 'pitch', 'semitones': 0.5},
                {'type': 'eq_shelf', 'freq': 1000, 'gain_db': 0.5}
            ]
        },
        'room_ambient': {
            'name': 'Room Ambience',
            'ops': [
                {'type': 'add_room', 'level_db': -30}
            ]
        }
    }
    
    def __init__(self):
        self.has_rubberband = self._check_rubberband()
        
    def _check_rubberband(self) -> bool:
        """Check if pyrubberband is available"""
        try:
            import pyrubberband
            return True
        except ImportError:
            logger.warning("pyrubberband not available, using librosa for pitch/time stretch")
            return False
    
    def generate_variant(
        self, 
        audio: np.ndarray, 
        sr: int,
        recipe_name: str
    ) -> Tuple[np.ndarray, Dict]:
        """Generate a voice variant using recipe"""
        
        if recipe_name not in self.RECIPES:
            raise ValueError(f"Unknown recipe: {recipe_name}")
        
        recipe = self.RECIPES[recipe_name]
        audio_out = audio.copy()
        
        # Track applied operations
        applied_ops = []
        
        for op in recipe['ops']:
            op_type = op['type']
            
            if op_type == 'rate':
                # Time stretch with formant preservation
                factor = op['factor']
                if self.has_rubberband:
                    import pyrubberband
                    audio_out = pyrubberband.time_stretch(audio_out, sr, factor)
                else:
                    audio_out = librosa.effects.time_stretch(audio_out, rate=factor)
                applied_ops.append(f"rate_{factor}")
                
            elif op_type == 'pitch':
                # Pitch shift with formant preservation
                semitones = op['semitones']
                # Clamp to safe range
                semitones = np.clip(semitones, -2, 2)
                
                if self.has_rubberband:
                    import pyrubberband
                    audio_out = pyrubberband.pitch_shift(audio_out, sr, semitones)
                else:
                    audio_out = librosa.effects.pitch_shift(
                        audio_out, sr=sr, n_steps=semitones
                    )
                applied_ops.append(f"pitch_{semitones}")
                
            elif op_type == 'eq_shelf':
                # Apply shelf EQ
                freq = op['freq']
                gain_db = op['gain_db']
                # Clamp gain
                gain_db = np.clip(gain_db, -3, 3)
                
                audio_out = self._apply_shelf_eq(audio_out, sr, freq, gain_db)
                applied_ops.append(f"eq_{freq}Hz_{gain_db}dB")
                
            elif op_type == 'add_room':
                # Add subtle room tone
                level_db = op['level_db']
                audio_out = self._add_room_tone(audio_out, sr, level_db)
                applied_ops.append(f"room_{level_db}dB")
        
        # Normalize to prevent clipping
        max_val = np.abs(audio_out).max()
        if max_val > 0.95:
            audio_out = audio_out * (0.95 / max_val)
        
        metadata = {
            'recipe': recipe_name,
            'recipe_name': recipe['name'],
            'operations': applied_ops
        }
        
        return audio_out, metadata
    
    def _apply_shelf_eq(
        self, 
        audio: np.ndarray, 
        sr: int,
        freq: float, 
        gain_db: float
    ) -> np.ndarray:
        """Apply high/low shelf EQ"""
        
        # Simple biquad shelf filter
        from scipy import signal
        
        # Normalize frequency
        w0 = 2 * np.pi * freq / sr
        
        # Shelf filter coefficients (simplified)
        A = 10**(gain_db / 40)
        
        if freq < 1000:  # Low shelf
            # Simplified low shelf
            b = [A, -A]
            a = [1, -(2-A)/(2+A)]
        else:  # High shelf
            # Simplified high shelf
            b = [A, A]
            a = [1, (2-A)/(2+A)]
        
        # Apply filter
        return signal.filtfilt(b, a, audio)
    
    def _add_room_tone(
        self, 
        audio: np.ndarray, 
        sr: int,
        level_db: float
    ) -> np.ndarray:
        """Add subtle room ambience"""
        
        # Generate pink noise at very low level
        duration = len(audio) / sr
        samples = len(audio)
        
        # Simple pink noise generation
        white = np.random.randn(samples)
        # Apply -3dB/octave filter (simplified)
        b = [0.04957526213389, -0.06305581334498, 0.01483220320740]
        a = [1.00000000000000, -1.80116083982126, 0.80257737639225]
        
        from scipy import signal
        pink = signal.lfilter(b, a, white)
        
        # Scale to desired level
        level_linear = 10**(level_db / 20)
        pink = pink * level_linear
        
        # Mix with original
        return audio + pink

# =============================================================================
# Voice Forge Main Class
# =============================================================================

class VoiceForge:
    """Main Voice Forge system for voice import and variant generation"""
    
    def __init__(self, base_dir: str = "demo/voices"):
        self.base_dir = base_dir
        self.quality_checker = VoiceQualityChecker()
        self.curator = VoiceCurator()
        self.variant_gen = VariantGenerator()
        
        # Ensure directories exist
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "imports"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "variants"), exist_ok=True)
    
    def import_voice(
        self, 
        audio_path: str,
        metadata: Dict[str, Any],
        auto_variants: bool = True
    ) -> Dict[str, Any]:
        """Import a voice with quality checking and curation"""
        
        # Check consent
        if not metadata.get('consent', False):
            raise ValueError("Consent required for voice import")
        
        # Check for celebrity names (basic denylist)
        name = metadata.get('name', '').lower()
        celebrity_keywords = [
            'obama', 'trump', 'biden', 'harris', 'musk', 'gates',
            'swift', 'kardashian', 'drake', 'beyonce', 'oprah'
        ]
        if any(celeb in name for celeb in celebrity_keywords):
            raise ValueError(f"Cannot use celebrity/public figure names")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Quality check
        quality = self.quality_checker.analyze(audio, sr)
        
        if not quality.passed:
            return {
                'success': False,
                'issues': quality.issues,
                'metrics': {
                    'duration': quality.duration,
                    'snr_db': quality.snr_db,
                    'rms_db': quality.rms_db,
                    'clipping_ratio': quality.clipping_ratio
                }
            }
        
        # Resample to 24kHz if needed
        if sr != 24000:
            logger.info(f"Resampling from {sr}Hz to 24000Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            sr = 24000
        
        # Find best segments
        segments = self.curator.find_best_segments(audio, sr)
        
        # Generate voice ID
        lang = metadata.get('lang', 'en')
        name = metadata.get('name', 'Custom')
        gender = metadata.get('gender', 'neutral')
        voice_id = f"{lang}-{name}_{gender}"
        
        # Ensure unique ID
        base_id = voice_id
        counter = 1
        while os.path.exists(os.path.join(self.base_dir, f"{voice_id}.wav")):
            voice_id = f"{base_id}_{counter}"
            counter += 1
        
        # Save main voice file
        voice_path = os.path.join(self.base_dir, "imports", f"{voice_id}.wav")
        
        # Normalize audio
        audio = audio / np.abs(audio).max() * 0.95
        
        # Save as WAV
        sf.write(voice_path, audio, sr)
        
        # Create sidecar metadata
        sidecar_data = {
            'id': voice_id,
            'lang': lang,
            'name': name,
            'gender': gender,
            'sr_hz': sr,
            'duration': float(len(audio) / sr),
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'duration': seg.duration,
                    'snr_db': seg.snr_db
                }
                for seg in segments
            ],
            'quality': {
                'snr_db': quality.snr_db,
                'rms_db': quality.rms_db,
                'clipping_ratio': quality.clipping_ratio,
                'speech_ratio': quality.vad_speech_ratio
            },
            'style': metadata.get('style', []),
            'notes': metadata.get('notes', ''),
            'consent': True,
            'import_timestamp': str(np.datetime64('now'))
        }
        
        sidecar_path = f"{voice_path}.yaml"
        with open(sidecar_path, 'w') as f:
            yaml.dump(sidecar_data, f, default_flow_style=False)
        
        # Generate variants if requested
        variants = []
        if auto_variants:
            variant_recipes = ['bright_fast', 'warm_slow', 'neutral_mid']
            for recipe in variant_recipes:
                try:
                    variant_id = self.create_variant(voice_id, recipe)
                    variants.append(variant_id)
                except Exception as e:
                    logger.warning(f"Failed to create {recipe} variant: {e}")
        
        return {
            'success': True,
            'voice_id': voice_id,
            'voice_path': voice_path,
            'variants': variants,
            'segments': len(segments),
            'quality': {
                'duration': quality.duration,
                'snr_db': quality.snr_db,
                'rms_db': quality.rms_db,
                'warnings': [iss for iss in quality.issues if 'Warning' in iss]
            }
        }
    
    def create_variant(self, base_voice_id: str, recipe_name: str) -> str:
        """Create a variant of existing voice"""
        
        # Remove .wav extension if present
        if base_voice_id.endswith('.wav'):
            base_voice_id = base_voice_id[:-4]
        
        # Find base voice
        base_path = None
        for subdir in ['', 'imports', 'variants']:
            test_path = os.path.join(self.base_dir, subdir, f"{base_voice_id}.wav")
            if os.path.exists(test_path):
                base_path = test_path
                break
        
        if not base_path:
            raise ValueError(f"Base voice not found: {base_voice_id}")
        
        # Load base voice
        audio, sr = librosa.load(base_path, sr=24000, mono=True)
        
        # Generate variant
        variant_audio, variant_metadata = self.variant_gen.generate_variant(
            audio, sr, recipe_name
        )
        
        # Create variant ID
        variant_id = f"{base_voice_id}_{recipe_name}"
        
        # Save variant
        variant_path = os.path.join(self.base_dir, "variants", f"{variant_id}.wav")
        sf.write(variant_path, variant_audio, sr)
        
        # Save variant metadata
        variant_sidecar = {
            'id': variant_id,
            'base_voice': base_voice_id,
            'recipe': recipe_name,
            'recipe_name': variant_metadata['recipe_name'],
            'operations': variant_metadata['operations'],
            'sr_hz': sr,
            'duration': float(len(variant_audio) / sr),
            'created': str(np.datetime64('now'))
        }
        
        with open(f"{variant_path}.yaml", 'w') as f:
            yaml.dump(variant_sidecar, f, default_flow_style=False)
        
        logger.info(f"Created variant: {variant_id}")
        return variant_id
    
    def list_available_recipes(self) -> Dict[str, str]:
        """List available variant recipes"""
        return {
            name: recipe['name'] 
            for name, recipe in VariantGenerator.RECIPES.items()
        }

# =============================================================================
# Testing
# =============================================================================

def test_voice_forge():
    """Test Voice Forge components"""
    
    print("\n" + "="*60)
    print("Testing Voice Forge")
    print("="*60)
    
    # Test 1: Quality checker
    print("\n[TEST 1] Quality Checker")
    checker = VoiceQualityChecker()
    
    # Create test audio (2 seconds at 24kHz)
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 48000))
    test_audio += np.random.randn(48000) * 0.01  # Add noise
    
    quality = checker.analyze(test_audio, 24000)
    print(f"Duration: {quality.duration:.1f}s")
    print(f"SNR: {quality.snr_db:.1f}dB")
    print(f"RMS: {quality.rms_db:.1f}dB")
    print(f"Passed: {quality.passed}")
    print(f"Issues: {quality.issues}")
    
    # Test 2: Variant generator
    print("\n[TEST 2] Variant Generator")
    gen = VariantGenerator()
    
    # List recipes
    print("Available recipes:")
    for name, desc in gen.RECIPES.items():
        print(f"  {name}: {desc['name']}")
    
    # Generate a variant
    variant_audio, metadata = gen.generate_variant(
        test_audio, 24000, 'bright_fast'
    )
    print(f"Variant generated: {metadata}")
    print(f"Original length: {len(test_audio)}")
    print(f"Variant length: {len(variant_audio)}")
    
    # Test 3: Voice curator
    print("\n[TEST 3] Voice Curator")
    curator = VoiceCurator()
    
    # Create longer test audio with varying quality
    long_audio = np.concatenate([
        np.sin(2 * np.pi * 440 * np.linspace(0, 10, 240000)),  # Clean
        np.random.randn(240000) * 0.5,  # Noise
        np.sin(2 * np.pi * 880 * np.linspace(0, 10, 240000))  # Clean high
    ])
    
    segments = curator.find_best_segments(long_audio, 24000, num_segments=3)
    print(f"Found {len(segments)} segments:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: {seg.start:.1f}-{seg.end:.1f}s, SNR={seg.snr_db:.1f}dB")
    
    print("\n✅ Voice Forge tests completed!")

if __name__ == "__main__":
    test_voice_forge()
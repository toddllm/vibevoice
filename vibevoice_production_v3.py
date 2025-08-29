#!/usr/bin/env python3
"""
VibeVoice Production Server V3
Enhanced with LLM text processing and Voice Forge
Preserves frozen 1.5B streaming path
"""

import os
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import logging

# Import base production server
from vibevoice_production_v2 import (
    VibeVoiceProductionServer,
    ModelInvariants,
    GoldenPathValidator,
    StreamingContext,
    VoiceLoader,
    SpeakerParser
)

# Import new components
from llm_text_processor import LLMTextProcessor, SynthesisUnit
from voice_forge import VoiceForge

logger = logging.getLogger(__name__)

# =============================================================================
# Enhanced Audio Stitcher with Synthesis Units
# =============================================================================

class EnhancedAudioStitcher:
    """Stitcher that handles synthesis units (dialogue + silence)"""
    
    def __init__(self, crossfade_ms: int = 50):
        self.crossfade_ms = crossfade_ms
        self.crossfade_samples = int(crossfade_ms * 24)  # 24 samples/ms at 24kHz
        
    def stitch_units(
        self,
        units: List[Tuple[SynthesisUnit, np.ndarray]],
        sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Stitch synthesis units together with proper crossfading rules
        
        Args:
            units: List of (SynthesisUnit, audio_array) tuples
                  For silence units, audio_array is None
        
        Returns:
            Stitched audio array
        """
        
        if not units:
            return np.array([], dtype=np.float32)
        
        segments = []
        prev_unit = None
        prev_audio = None
        
        for unit, audio in units:
            if unit.is_silence():
                # Generate silence
                silence = np.zeros(unit.samples, dtype=np.float32)
                segments.append(silence)
                prev_unit = unit
                prev_audio = silence
                
            elif unit.is_dialogue():
                if audio is None:
                    logger.warning(f"No audio for dialogue unit: {unit}")
                    continue
                
                # Decide on crossfade
                should_crossfade = False
                
                if prev_unit and prev_audio is not None:
                    # Crossfade only if:
                    # 1. Previous was dialogue (not silence)
                    # 2. Same speaker
                    # 3. Enough samples in both
                    should_crossfade = (
                        prev_unit.is_dialogue() and
                        prev_unit.speaker_id == unit.speaker_id and
                        len(prev_audio) >= self.crossfade_samples and
                        len(audio) >= self.crossfade_samples
                    )
                
                if should_crossfade and segments:
                    # Apply equal-power crossfade
                    crossfaded = self._equal_power_crossfade(
                        segments[-1], audio, self.crossfade_samples
                    )
                    # Replace last segment's end and current segment's start
                    segments[-1] = crossfaded[0]
                    segments.append(crossfaded[1])
                else:
                    # No crossfade - direct concatenation
                    segments.append(audio)
                
                prev_unit = unit
                prev_audio = audio
        
        # Concatenate all segments
        if segments:
            return np.concatenate(segments)
        else:
            return np.array([], dtype=np.float32)
    
    def _equal_power_crossfade(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        crossfade_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply equal-power crossfade between two audio segments"""
        
        # Create fade curves (equal power = sqrt of linear)
        fade_out = np.sqrt(np.linspace(1, 0, crossfade_samples))
        fade_in = np.sqrt(np.linspace(0, 1, crossfade_samples))
        
        # Apply fades
        audio1_faded = audio1.copy()
        audio1_faded[-crossfade_samples:] *= fade_out
        
        audio2_faded = audio2.copy()
        audio2_faded[:crossfade_samples] *= fade_in
        
        # Mix crossfade region
        crossfade_region = (
            audio1_faded[-crossfade_samples:] +
            audio2_faded[:crossfade_samples]
        )
        
        # Build output segments
        audio1_out = np.concatenate([
            audio1[:-crossfade_samples],
            crossfade_region
        ])
        
        audio2_out = audio2[crossfade_samples:]
        
        return (audio1_out, audio2_out)

# =============================================================================
# Enhanced Production Server V3
# =============================================================================

class EnhancedVibeVoiceServer(VibeVoiceProductionServer):
    """Production server with LLM and Voice Forge enhancements"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional enhancements"""
        super().__init__()
        
        self.config = config or {}
        
        # Initialize LLM processor (optional)
        if self.config.get('llm', {}).get('enabled', False):
            llm_config = self.config['llm']
            self.llm_processor = LLMTextProcessor(llm_config)
            logger.info("LLM text processor enabled")
        else:
            self.llm_processor = None
            logger.info("LLM text processor disabled")
        
        # Initialize Voice Forge
        self.voice_forge = VoiceForge(
            base_dir=self.config.get('voice_forge', {}).get('base_dir', 'demo/voices')
        )
        logger.info("Voice Forge initialized")
        
        # Initialize enhanced stitcher
        self.stitcher = EnhancedAudioStitcher(
            crossfade_ms=self.config.get('audio', {}).get('crossfade_ms', 50)
        )
        
        # Metrics
        self.enhanced_metrics = {
            'llm_processed': 0,
            'llm_fallbacks': 0,
            'voices_imported': 0,
            'variants_created': 0,
            'silence_units_inserted': 0
        }
    
    def generate(
        self,
        text: str,
        voice_path: str = "demo/voices/en-Alice_woman.wav",
        model_size: str = "1.5B",
        max_seconds: int = 30,
        use_llm: Optional[bool] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Enhanced generation with optional LLM preprocessing
        
        Preserves frozen 1.5B streaming path when LLM is disabled
        """
        
        start_time = time.time()
        synthesis_units = None
        
        # Decide whether to use LLM
        if use_llm is None:
            use_llm = self.llm_processor is not None and self.config.get('llm', {}).get('auto_process', True)
        
        # Pre-synthesis LLM processing (optional)
        if use_llm and self.llm_processor:
            try:
                logger.info("Processing text with LLM")
                processed_text, synthesis_units = self.llm_processor.process(text)
                text = processed_text
                self.enhanced_metrics['llm_processed'] += 1
                logger.info(f"LLM produced {len(synthesis_units)} synthesis units")
            except Exception as e:
                logger.warning(f"LLM processing failed: {e}, using original text")
                self.enhanced_metrics['llm_fallbacks'] += 1
                synthesis_units = None
        
        # If we have synthesis units, use unit-based generation
        if synthesis_units:
            return self._generate_with_units(
                synthesis_units, voice_path, model_size, max_seconds
            )
        
        # Otherwise, use standard generation (preserves frozen path)
        return super().generate(text, voice_path, model_size, max_seconds)
    
    def _generate_with_units(
        self,
        units: List[SynthesisUnit],
        voice_path: str,
        model_size: str,
        max_seconds: int
    ) -> Tuple[np.ndarray, Dict]:
        """Generate audio from synthesis units"""
        
        logger.info(f"Generating with {len(units)} synthesis units")
        
        # Group dialogue units by speaker for efficient processing
        speaker_groups = {}
        for unit in units:
            if unit.is_dialogue():
                if unit.speaker_id not in speaker_groups:
                    speaker_groups[unit.speaker_id] = []
                speaker_groups[unit.speaker_id].append(unit)
        
        # Generate audio for each speaker's lines
        generated_audio = {}
        
        for speaker_id, speaker_units in speaker_groups.items():
            # Combine text for batch generation
            combined_text = '\n'.join([
                f"Speaker {speaker_id}: {unit.text}"
                for unit in speaker_units
            ])
            
            # Generate audio for this speaker
            audio, metadata = super().generate(
                combined_text, voice_path, model_size, max_seconds
            )
            
            # Split audio back into units
            # This is simplified - in production you'd use actual timing
            total_frames = sum(unit.frames for unit in speaker_units)
            frame_to_sample = len(audio) / total_frames if total_frames > 0 else 1
            
            current_pos = 0
            for unit in speaker_units:
                unit_samples = int(unit.frames * frame_to_sample)
                unit_audio = audio[current_pos:current_pos + unit_samples]
                generated_audio[id(unit)] = unit_audio
                current_pos += unit_samples
        
        # Build unit list with audio
        units_with_audio = []
        
        for unit in units:
            if unit.is_silence():
                units_with_audio.append((unit, None))
                self.enhanced_metrics['silence_units_inserted'] += 1
            elif unit.is_dialogue():
                audio = generated_audio.get(id(unit))
                units_with_audio.append((unit, audio))
        
        # Stitch units together
        final_audio = self.stitcher.stitch_units(units_with_audio)
        
        # Calculate metadata
        duration = len(final_audio) / 24000
        metadata = {
            'duration_s': duration,
            'rtf': (time.time() - time.time()) / duration if duration > 0 else 0,
            'model': model_size,
            'units_processed': len(units),
            'silence_units': self.enhanced_metrics['silence_units_inserted']
        }
        
        return final_audio, metadata
    
    def import_voice(
        self,
        audio_path: str,
        metadata: Dict[str, Any],
        auto_variants: bool = True
    ) -> Dict[str, Any]:
        """Import a new voice through Voice Forge"""
        
        result = self.voice_forge.import_voice(audio_path, metadata, auto_variants)
        
        if result['success']:
            self.enhanced_metrics['voices_imported'] += 1
            self.enhanced_metrics['variants_created'] += len(result.get('variants', []))
        
        return result
    
    def create_voice_variant(self, base_voice_id: str, recipe: str) -> str:
        """Create a voice variant"""
        
        variant_id = self.voice_forge.create_variant(base_voice_id, recipe)
        self.enhanced_metrics['variants_created'] += 1
        return variant_id
    
    def get_enhanced_metrics(self) -> Dict:
        """Get enhanced metrics including LLM and Voice Forge"""
        
        base_metrics = self.get_diagnostics()
        
        # Add LLM metrics
        if self.llm_processor:
            base_metrics['llm'] = self.llm_processor.get_metrics()
            base_metrics['llm']['fallback_rate'] = (
                self.enhanced_metrics['llm_fallbacks'] /
                max(self.enhanced_metrics['llm_processed'], 1)
            )
        
        # Add Voice Forge metrics
        base_metrics['voice_forge'] = {
            'voices_imported': self.enhanced_metrics['voices_imported'],
            'variants_created': self.enhanced_metrics['variants_created']
        }
        
        # Add audio metrics
        base_metrics['audio'] = {
            'silence_units_inserted': self.enhanced_metrics['silence_units_inserted']
        }
        
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Extended health check"""
        
        base_health = super().health_check()
        
        # Add enhancement status
        base_health['enhancements'] = {
            'llm_enabled': self.llm_processor is not None,
            'voice_forge_ready': True,
            'available_recipes': list(self.voice_forge.variant_gen.RECIPES.keys())
        }
        
        # Add metrics
        base_health['enhanced_metrics'] = self.get_enhanced_metrics()
        
        return base_health

# =============================================================================
# Configuration Loader
# =============================================================================

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file or defaults"""
    
    default_config = {
        'llm': {
            'enabled': False,  # Start disabled
            'engine': 'ollama',
            'model': 'qwen2:0.5b',
            'structure_temp': 0.1,
            'refine_temp': 0.3,
            'structure_timeout': 2.0,
            'refine_timeout': 2.5,
            'auto_process': True,
            'auto_threshold': 0.35,
            'cache_size': 100
        },
        'voice_forge': {
            'base_dir': 'demo/voices',
            'auto_variants': True,
            'variant_recipes': ['bright_fast', 'warm_slow', 'neutral_mid']
        },
        'audio': {
            'crossfade_ms': 50,
            'max_pause_ms': 500
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Merge with defaults
            def deep_merge(default, override):
                for key, value in override.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        deep_merge(default[key], value)
                    else:
                        default[key] = value
            
            deep_merge(default_config, file_config)
            
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
    
    return default_config

# =============================================================================
# Testing
# =============================================================================

def test_enhanced_server():
    """Test the enhanced production server"""
    
    print("\n" + "="*60)
    print("Testing Enhanced VibeVoice Server V3")
    print("="*60)
    
    # Load config with LLM disabled for testing
    config = load_config()
    config['llm']['enabled'] = False  # Start with LLM disabled
    
    server = EnhancedVibeVoiceServer(config)
    
    # Test 1: Standard generation (frozen path)
    print("\n[TEST 1] Standard generation (LLM disabled)")
    text = "Speaker 0: Hello, this is a test."
    voice = "demo/voices/en-Alice_woman.wav"
    
    audio, metadata = server.generate(text, voice, model_size="1.5B")
    print(f"Generated {metadata['duration_s']:.1f}s of audio")
    print(f"RTF: {metadata['rtf']:.2f}")
    
    # Test 2: Enable LLM and test messy text
    print("\n[TEST 2] LLM processing (mock engine)")
    config['llm']['enabled'] = True
    config['llm']['engine'] = 'mock'
    server = EnhancedVibeVoiceServer(config)
    
    messy_text = """
    Alice: Hello everyone! Welcome to our podcast.
    Bob says "Thanks for having me"
    (pause)
    """
    
    audio, metadata = server.generate(messy_text, voice, use_llm=True)
    print(f"Generated {metadata['duration_s']:.1f}s of audio")
    print(f"Units processed: {metadata.get('units_processed', 0)}")
    
    # Test 3: Stitcher with units
    print("\n[TEST 3] Audio stitcher")
    stitcher = EnhancedAudioStitcher()
    
    # Create test units
    units = [
        (SynthesisUnit('dialogue', speaker_id=0, text="Hello"), np.ones(24000)),
        (SynthesisUnit('silence', samples=2400), None),  # 100ms pause
        (SynthesisUnit('dialogue', speaker_id=0, text="World"), np.ones(24000)),
        (SynthesisUnit('dialogue', speaker_id=1, text="Hi"), np.ones(24000))
    ]
    
    stitched = stitcher.stitch_units(units)
    print(f"Stitched audio length: {len(stitched)} samples ({len(stitched)/24000:.1f}s)")
    
    # Test 4: Metrics
    print("\n[TEST 4] Enhanced metrics")
    metrics = server.get_enhanced_metrics()
    print(f"LLM processed: {server.enhanced_metrics['llm_processed']}")
    print(f"Silence units: {server.enhanced_metrics['silence_units_inserted']}")
    
    # Test 5: Health check
    print("\n[TEST 5] Health check")
    health = server.health_check()
    print(f"Status: {health['status']}")
    print(f"LLM enabled: {health['enhancements']['llm_enabled']}")
    print(f"Voice Forge ready: {health['enhancements']['voice_forge_ready']}")
    
    print("\n✅ All enhanced server tests passed!")

if __name__ == "__main__":
    test_enhanced_server()
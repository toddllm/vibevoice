#!/usr/bin/env python3
"""
Phi4-Only Text Processor for VibeVoice
Pure LLM pipeline with server compatibility layer - NO FALLBACKS
"""

import re
import json
import hashlib
import time
import os
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def _call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Call function with timeout using thread pool"""
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_s}s")

@dataclass
class SynthesisUnit:
    """Single unit for TTS synthesis"""
    unit_type: str  # 'dialogue', 'narration', 'direction' 
    speaker_id: int
    text: str
    frames: int
    pause_ms_before: int = 0
    pause_ms_after: int = 0
    
    def is_dialogue(self) -> bool:
        """Check if this unit is dialogue - LLM decides all formatting"""
        return True  # Simplified: let TTS handle all text as dialogue
        
    def is_silence(self) -> bool:
        """Check if this unit is silence"""
        return False  # No silence units in our simple design

class FrameBudgeter:
    """Calculate frame estimates for synthesis units"""
    
    def __init__(self):
        self.chars_per_second = 12  # Conservative estimate
        self.sample_rate = 24000
        
    def calculate_frames(self, text: str) -> int:
        """Estimate frames needed for text"""
        # Rough calculation: chars -> seconds -> frames
        char_count = len(text)
        seconds = char_count / self.chars_per_second
        frames = int(seconds * self.sample_rate)
        return max(frames, 1000)  # Minimum 1000 frames

class LLMEngine:
    """Base LLM engine interface"""
    
    def generate(self, prompt: str, temperature: float = 0.0, timeout: Optional[float] = None) -> str:
        raise NotImplementedError

class OllamaEngine(LLMEngine):
    """Ollama engine for phi4-mini-reasoning"""
    
    def __init__(self, model: str = "phi4-mini-reasoning:3.8b"):
        self.model = model
        self._init_client()
        
    def _init_client(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self.client = ollama.Client()
            logger.info(f"Initializing Ollama with model: {self.model}")
            
            # Verify model exists
            try:
                self.client.show(self.model)
            except Exception as e:
                logger.warning(f"Model {self.model} not found, pulling...")
                self.client.pull(self.model)
                
        except ImportError:
            logger.error("Ollama package not found. Install with: pip install ollama")
            raise
            
    def generate(self, prompt: str, temperature: float = 0.0, timeout: Optional[float] = None) -> str:
        """Generate response using Ollama"""
        try:
            if timeout:
                return _call_with_timeout(self._generate_sync, timeout, prompt, temperature)
            else:
                return self._generate_sync(prompt, temperature)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
            
    def _generate_sync(self, prompt: str, temperature: float) -> str:
        """Synchronous generation call"""
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        
        content = response.get('message', {}).get('content', '')
        logger.info(f"LLM response length: {len(content)} chars")
        return content

class MockEngine(LLMEngine):
    """Mock engine for testing"""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["Speaker 0: Mock response"]
        self.call_count = 0
        
    def generate(self, prompt: str, temperature: float = 0.0, timeout: Optional[float] = None) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

class LLMTextProcessor:
    """
    LLM-only text processor with server compatibility.
    All processing goes through LLM (qwen3:1.7b) - NO FALLBACKS.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize LLM processor"""
        if not config:
            config = {
                'engine': 'ollama',
                'model': 'qwen3:8b',
                'structure_timeout': 5 * 60  # 5 minutes for qwen3
            }
        
        self.config = config
        self.timeout_s = config.get('structure_timeout', 120)  # 2 minute timeout for complex formats
        self.engine = self._init_engine(config)
        
        # Live metrics - no fake numbers
        self._ok = False
        self._last_err = None
        self._req_count = 0
        self._lat_accum = 0.0
        self._model_name = config.get('model', 'phi4-mini-reasoning:3.8b')
        
        # Initialize supporting components
        self.budgeter = FrameBudgeter()
        
        # Initialize voice mapper for intelligent voice assignment
        from voice_mapper import VoiceMapper
        self.voice_mapper = VoiceMapper()
        
        logger.info(f"LLMTextProcessor initialized: {self._model_name}")
        
    def _init_engine(self, config: Dict) -> LLMEngine:
        """Initialize the appropriate LLM engine"""
        engine_type = config.get('engine', 'ollama')
        model = config.get('model', 'phi4-mini-reasoning:3.8b')
        
        if engine_type == 'ollama':
            return OllamaEngine(model)
        elif engine_type == 'mock':
            return MockEngine()
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
    
    # === PRIMARY API ===
    def process(self, text: str, force_llm: bool = True, return_units: bool = True) -> Tuple[str, Optional[List[SynthesisUnit]]]:
        """
        LLM-only pipeline: text → phi4 → VibeVoice format → synthesis units
        """
        t0 = time.time()
        try:
            # Run phi4 pipeline
            formatted_text, units = self._phi4_pipeline(text, want_units=return_units)
            
            # Update metrics
            self._ok = True
            self._req_count += 1
            self._lat_accum += (time.time() - t0)
            
            return formatted_text, (units if return_units else None)
            
        except Exception as e:
            self._ok = False
            self._last_err = str(e)
            logger.error(f"Phi4 processing failed: {e}")
            raise
    
    # === SERVER COMPATIBILITY LAYER ===
    def _normalize_text(self, text: str) -> str:
        """
        Server compatibility: LLM-only normalization
        Delegates to process() to maintain single source of truth
        """
        formatted, _ = self.process(text, return_units=False)
        return formatted
    
    def get_metrics(self) -> Dict:
        """
        Server compatibility: return real metrics
        """
        avg_latency = (self._lat_accum / self._req_count) if self._req_count else None
        return {
            "ok": self._ok,
            "model": self._model_name,
            "requests": self._req_count,
            "avg_latency_s": avg_latency,
            "last_error": self._last_err,
        }
    
    # === INTERNAL PHI4 PIPELINE ===
    def _phi4_pipeline(self, text: str, want_units: bool) -> Tuple[str, Optional[List[SynthesisUnit]]]:
        """
        Core phi4 processing pipeline
        """
        logger.info("STARTING LLM PROCESSING")
        
        # Use shared prompt template
        from prompt_templates import VibeVoicePrompts
        prompt = VibeVoicePrompts.get_current_prompt(text)

        try:
            # Generate with phi4
            content = self.engine.generate(
                prompt,
                temperature=0.0,
                timeout=self.timeout_s
            )
            
            if not content:
                raise RuntimeError("LLM returned empty response")
                
            logger.info(f"LLM raw response: {content[:300]}...")
            
            # Parse LLM response to extract VibeVoice format
            formatted_text = self._extract_vibekvoice_format(content)
            
            # Generate synthesis units if requested
            units = None
            if want_units:
                units = self._create_synthesis_units(formatted_text)
                
                # Apply intelligent voice assignment in post-processing
                voice_assignments = self._assign_voices_to_speakers(units, text)
                if voice_assignments:
                    logger.info(f"Voice assignments: {voice_assignments}")
                    self._last_voice_assignments = voice_assignments
            
            logger.info(f"LLM success: {len(units or [])} synthesis units")
            return formatted_text, units
            
        except Exception as e:
            logger.error(f"LLM pipeline failed: {e}")
            raise RuntimeError(f"LLM processing failed: {e}")
    
    def _extract_vibekvoice_format(self, content: str) -> str:
        """Extract VibeVoice format lines from LLM response"""
        logger.info(f"Parsing LLM response: {content[:500]}...")
        
        # Remove thinking blocks
        content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Remove code blocks but preserve their content if it contains Speaker lines
        def extract_from_code_block(match):
            code_content = match.group(0)[3:-3]  # Remove ```
            if 'Speaker' in code_content:
                return code_content
            return ''
        content_clean = re.sub(r'```.*?```', extract_from_code_block, content_clean, flags=re.DOTALL)
        
        # Find lines that match "Speaker N: text" pattern and extract JSON voice mapping
        speaker_lines = []
        voice_assignments = {}
        
        # Look for JSON voice mapping in response
        json_match = re.search(r'\{.*"voice_assignments".*\}', content_clean, re.DOTALL)
        if json_match:
            try:
                voice_mapping = json.loads(json_match.group())
                if 'voice_assignments' in voice_mapping:
                    # Convert string keys to integers
                    voice_assignments = {int(k): v for k, v in voice_mapping['voice_assignments'].items()}
                    logger.info(f"Extracted voice assignments from LLM: {voice_assignments}")
            except json.JSONDecodeError:
                logger.warning("Failed to parse voice assignments JSON from LLM")
        
        # Extract speaker lines
        for line in content_clean.split('\n'):
            line = line.strip()
            if re.match(r'^Speaker \d+:', line):
                speaker_lines.append(line)
        
        # If we didn't find Speaker format, look for other common patterns phi4 might use
        if not speaker_lines:
            logger.info("No 'Speaker N:' lines found, looking for alternative formats...")
            
            # Look for any line with speaker-like patterns
            alternative_patterns = [
                r'^[A-Z][a-z]+\s*:',  # "Alice:", "Narrator:"
                r'^\d+\.\s*Speaker',   # "1. Speaker 1:"
                r'^-\s*Speaker',       # "- Speaker 1:"
            ]
            
            for pattern in alternative_patterns:
                for line in content_clean.split('\n'):
                    line = line.strip()
                    if re.match(pattern, line):
                        # Convert to Speaker N: format
                        if 'Speaker' not in line:
                            # Convert "Alice: text" to "Speaker 1: text" 
                            converted = re.sub(r'^([A-Z][a-z]+)\s*:', r'Speaker 1:', line)
                            speaker_lines.append(converted)
                        else:
                            speaker_lines.append(line)
                
                if speaker_lines:
                    break
        
        if not speaker_lines:
            # Log the full response for debugging
            logger.error(f"Full LLM response (first 1000 chars): {content[:1000]}")
            logger.error(f"Content after cleaning (first 1000 chars): {content_clean[:1000]}")
            raise RuntimeError("No VibeVoice format lines found in LLM response")
        
        logger.info(f"Found {len(speaker_lines)} speaker lines: {speaker_lines[:3]}")
        if voice_assignments:
            logger.info(f"Voice assignments: {voice_assignments}")
            # Store voice assignments for synthesis use
            self._last_voice_assignments = voice_assignments
        
        return '\n'.join(speaker_lines)
    
    def _create_synthesis_units(self, formatted_text: str) -> List[SynthesisUnit]:
        """Convert formatted text to synthesis units - let LLM decide dialogue vs narration"""
        units = []
        
        for line in formatted_text.split('\n'):
            if line.strip():
                match = re.match(r'Speaker (\d+): (.+)', line.strip())
                if match:
                    speaker_id = int(match.group(1))
                    text = match.group(2)
                    frames = self.budgeter.calculate_frames(text)
                    
                    # Simple: everything is dialogue since LLM already decided the format
                    units.append(SynthesisUnit(
                        unit_type='dialogue',
                        speaker_id=speaker_id,
                        text=text,
                        frames=frames
                    ))
        
        return units
    
    def _assign_voices_to_speakers(self, units: List[SynthesisUnit], original_text: str) -> Dict[int, str]:
        """Intelligently assign voices to speakers based on character analysis"""
        if not units:
            return {}
        
        # Extract character names from original text
        character_names = self._extract_character_names(original_text)
        
        # Map speaker IDs to character names
        speaker_to_character = {}
        character_index = 0
        
        for unit in units:
            if unit.speaker_id > 0 and unit.speaker_id not in speaker_to_character:
                if character_index < len(character_names):
                    speaker_to_character[unit.speaker_id] = character_names[character_index]
                    character_index += 1
                else:
                    speaker_to_character[unit.speaker_id] = f"Character{unit.speaker_id}"
        
        # Use voice mapper to assign appropriate voices
        voice_assignments = self.voice_mapper.assign_voices_to_speakers(speaker_to_character)
        
        return voice_assignments
    
    def _extract_character_names(self, text: str) -> List[str]:
        """Extract character names from original text"""
        # Look for common patterns: "Alice:", "Bob says", "said Alice", etc.
        names = []
        
        # Pattern 1: "Name:" at start of line
        for match in re.finditer(r'^(\w+):', text, re.MULTILINE):
            name = match.group(1)
            if name not in names and name not in ['Speaker', 'INT', 'EXT']:
                names.append(name)
        
        # Pattern 2: "Name says", "said Name"
        for match in re.finditer(r'\b(\w+)\s+(says?|said|replied?|responds?)', text):
            name = match.group(1)
            if name not in names and len(name) > 2:
                names.append(name)
        
        for match in re.finditer(r'\b(says?|said|replied?|responds?)\s+(\w+)', text):
            name = match.group(2)
            if name not in names and len(name) > 2:
                names.append(name)
        
        return names[:5]  # Limit to 5 characters to avoid confusion
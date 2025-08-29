#!/usr/bin/env python3
"""
LLM Text Processing Pipeline for VibeVoice
Precise, low-risk design with messiness scoring and two-pass processing
"""

import re
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
import logging
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SynthesisUnit:
    """Unit of synthesis (dialogue or silence)"""
    unit_type: str  # 'dialogue' or 'silence'
    speaker_id: Optional[int] = None
    text: Optional[str] = None
    frames: Optional[int] = None
    samples: Optional[int] = None  # For silence units
    
    def is_silence(self) -> bool:
        return self.unit_type == 'silence'
    
    def is_dialogue(self) -> bool:
        return self.unit_type == 'dialogue'

# =============================================================================
# Messiness Scorer
# =============================================================================

class MessinessScorer:
    """Scores text messiness to decide LLM vs regex path"""
    
    def score(self, text: str) -> float:
        """
        Calculate messiness score (0-1)
        >= 0.35 triggers LLM processing
        """
        score = 0.0
        lines = text.strip().split('\n')
        
        # Signal 1: Lines without speaker tags (weight: 0.25)
        no_speaker_lines = sum(1 for line in lines if line.strip() and ':' not in line)
        if lines:
            score += (no_speaker_lines / len(lines)) * 0.25
        
        # Signal 2: Mixed formats (named + Speaker N:) (weight: 0.20)
        has_named = any(re.match(r'^[A-Za-z]+:', line) for line in lines)
        has_speaker_n = any(re.match(r'^Speaker \d+:', line) for line in lines)
        if has_named and has_speaker_n:
            score += 0.20
        
        # Signal 3: Stage directions (weight: 0.15)
        stage_pattern = r'\([^)]+\)|\[[^\]]+\]|\{[^}]+\}'
        if any(re.search(stage_pattern, line) for line in lines):
            score += 0.15
        
        # Signal 4: Long lines or complex punctuation (weight: 0.10)
        long_lines = sum(1 for line in lines if len(line) > 150 or line.count(',') >= 3)
        if lines and long_lines > len(lines) * 0.3:
            score += 0.10
        
        # Signal 5: Bullet lists / Markdown (weight: 0.10)
        markdown_indicators = [r'^\s*[-*+]\s', r'^\s*\d+\.\s', r'^#{1,6}\s', r'^\s*>']
        if any(re.match(pattern, line) for line in lines for pattern in markdown_indicators):
            score += 0.10
        
        # Signal 6: Multiple blank lines (weight: 0.10)
        blank_runs = re.findall(r'\n\n+', text)
        if any(len(run) > 2 for run in blank_runs):
            score += 0.10
        
        # Signal 7: Many contractions or curly quotes (weight: 0.10)
        contractions = re.findall(r"\w+[''](?:t|s|re|ve|ll|d|m)\b", text, re.IGNORECASE)
        curly_quotes = text.count('"') + text.count('"') + text.count(''') + text.count(''')
        if len(contractions) > len(lines) * 3 or curly_quotes > 5:
            score += 0.10
        
        return min(score, 1.0)

# =============================================================================
# JSON Schema and Validation
# =============================================================================

DIALOGUE_SCHEMA_V1 = {
    "type": "object",
    "properties": {
        "version": {"const": "vv-1"},
        "language": {"type": "string"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "speaker": {"type": "string"},
                    "type": {"enum": ["dialogue", "narration", "direction"]},
                    "text": {"type": "string"},
                    "lang": {"type": "string"},
                    "line_frames": {"type": "integer"},
                    "cues": {
                        "type": "object",
                        "properties": {
                            "pause_ms_before": {"type": "integer", "maximum": 500},
                            "pause_ms_after": {"type": "integer", "maximum": 500},
                            "style": {"type": "array"},
                            "emphasis_words": {"type": "array"}
                        }
                    }
                },
                "required": ["id", "speaker", "type", "text"]
            }
        }
    },
    "required": ["version", "lines"]
}

class SchemaGuard:
    """Validates and repairs JSON output from LLM"""
    
    def validate_and_repair(self, data: Any, max_repairs: int = 1) -> Optional[Dict]:
        """Validate JSON against schema with limited repair attempts"""
        
        # First ensure it's a dict
        if not isinstance(data, dict):
            logger.warning(f"Invalid data type: {type(data)}")
            return None
        
        repairs_made = 0
        
        # Check version
        if 'version' not in data:
            data['version'] = 'vv-1'
            repairs_made += 1
        
        # Check lines array
        if 'lines' not in data or not isinstance(data['lines'], list):
            logger.warning("Missing or invalid lines array")
            return None
        
        # Repair each line
        for i, line in enumerate(data['lines']):
            if not isinstance(line, dict):
                logger.warning(f"Line {i} is not a dict")
                return None
            
            # Ensure required fields
            if 'id' not in line:
                line['id'] = i
                repairs_made += 1
            
            if 'speaker' not in line or not line['speaker']:
                line['speaker'] = 'Narrator'
                repairs_made += 1
            
            if 'type' not in line:
                line['type'] = 'dialogue'
                repairs_made += 1
            
            if 'text' not in line or not line['text']:
                logger.warning(f"Line {i} has no text")
                return None
            
            # Add language if missing
            if 'lang' not in line:
                line['lang'] = data.get('language', 'en')
        
        if repairs_made > max_repairs:
            logger.warning(f"Too many repairs needed: {repairs_made}")
            return None
        
        return data

# =============================================================================
# Frame Budgeting
# =============================================================================

class FrameBudgeter:
    """Calculate frame-accurate budgets for synthesis"""
    
    WPS = 2.75  # Words per second (calibrated)
    FRAME_RATE = 7.5  # Frames per second
    
    def calculate_frames(self, text: str) -> int:
        """Calculate frame budget for text"""
        # Count words (including contractions)
        words = len(re.findall(r"\w+[''']?\w*", text))
        seconds = words / self.WPS
        frames = round(seconds * self.FRAME_RATE)
        return max(frames, 1)  # At least 1 frame
    
    def add_frame_budgets(self, data: Dict) -> Dict:
        """Add line_frames to each line in JSON"""
        for line in data.get('lines', []):
            if 'text' in line:
                line['line_frames'] = self.calculate_frames(line['text'])
        return data

# =============================================================================
# LLM Engine Interface
# =============================================================================

class LLMEngine:
    """Base class for LLM backends"""
    
    def generate(self, prompt: str, temperature: float = 0.1, timeout: float = 2.0) -> Optional[str]:
        raise NotImplementedError

class OllamaEngine(LLMEngine):
    """Ollama backend for local LLM"""
    
    def __init__(self, model: str = "qwen2:0.5b"):
        self.model = model
        # Import will be done lazily when needed
        self.client = None
    
    def _init_client(self):
        """Lazy initialization of Ollama client"""
        if self.client is None:
            try:
                import ollama
                self.client = ollama.Client()
            except ImportError:
                logger.error("Ollama not installed. Install with: pip install ollama")
                raise
    
    def generate(self, prompt: str, temperature: float = 0.1, timeout: float = 2.0) -> Optional[str]:
        """Generate with timeout and error handling"""
        self._init_client()
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"LLM generation timed out after {timeout}s")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                format="json",
                options={"num_predict": 2000}
            )
            
            # Cancel timeout
            signal.alarm(0)
            
            return response.get('response', '')
            
        except TimeoutError as e:
            logger.warning(str(e))
            return None
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None

class MockEngine(LLMEngine):
    """Mock engine for testing"""
    
    def generate(self, prompt: str, temperature: float = 0.1, timeout: float = 2.0) -> Optional[str]:
        """Return mock structured response"""
        if "structure" in prompt.lower():
            return json.dumps({
                "version": "vv-1",
                "language": "en",
                "lines": [
                    {"id": 0, "speaker": "Alice", "type": "dialogue", "text": "Hello everyone!", "lang": "en"},
                    {"id": 1, "speaker": "Bob", "type": "dialogue", "text": "Hi Alice!", "lang": "en"}
                ]
            })
        else:  # Refine pass
            return json.dumps({
                "version": "vv-1",
                "language": "en",
                "lines": [
                    {"id": 0, "speaker": "Alice", "type": "dialogue", "text": "Hello, everyone!", "lang": "en", 
                     "cues": {"pause_ms_after": 200}},
                    {"id": 1, "speaker": "Bob", "type": "dialogue", "text": "Hi, Alice!", "lang": "en"}
                ]
            })

# =============================================================================
# Two-Pass LLM Processor
# =============================================================================

class LLMTextProcessor:
    """Main LLM text processing pipeline"""
    
    # System prompts
    STRUCTURE_PROMPT = """You convert messy text into a structured dialogue JSON for TTS. Don't invent content. Don't merge or split speakers unless obvious. Each utterance is short (≤ 25 words). Return ONLY valid JSON that matches the schema."""
    
    REFINE_PROMPT = """You lightly polish each utterance for TTS. Keep the *same array length and id values*. Normalize apostrophes. Fix obvious transcription errors. Optionally add pause_ms_before/after (≤ 500). Return ONLY valid JSON with the **same ids**."""
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        # Components
        self.scorer = MessinessScorer()
        self.guard = SchemaGuard()
        self.budgeter = FrameBudgeter()
        
        # LLM engine
        engine_type = config.get('engine', 'mock')
        if engine_type == 'ollama':
            self.engine = OllamaEngine(config.get('model', 'qwen2:0.5b'))
        else:
            self.engine = MockEngine()
        
        # Config
        self.auto_threshold = config.get('auto_threshold', 0.35)
        self.structure_temp = config.get('structure_temp', 0.1)
        self.refine_temp = config.get('refine_temp', 0.3)
        self.structure_timeout = config.get('structure_timeout', 2.0)
        self.refine_timeout = config.get('refine_timeout', 2.5)
        
        # Cache
        self.cache = OrderedDict()
        self.max_cache_size = config.get('cache_size', 100)
        
        # Metrics
        self.metrics = {
            'llm_calls': 0,
            'fallbacks': 0,
            'cache_hits': 0,
            'avg_latency': 0.0
        }
    
    def process(self, text: str, force_llm: bool = False) -> Tuple[str, List[SynthesisUnit]]:
        """
        Process text and return both formatted text and synthesis units
        Returns: (formatted_text, synthesis_units)
        """
        
        # Normalize input
        text = self._normalize_text(text)
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            self.metrics['cache_hits'] += 1
            self.cache.move_to_end(cache_key)  # LRU update
            return self.cache[cache_key]
        
        # Calculate messiness score
        messiness = self.scorer.score(text)
        logger.info(f"Text messiness score: {messiness:.2f}")
        
        # Decide processing path
        if not force_llm and messiness < self.auto_threshold:
            # Use fast regex path
            result = self._regex_fallback(text)
        else:
            # Use LLM path
            result = self._llm_process(text)
            if result is None:
                # Fallback on LLM failure
                result = self._regex_fallback(text)
                self.metrics['fallbacks'] += 1
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (unicode, whitespace, etc.)"""
        # Normalize unicode quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove code fences
        text = re.sub(r'```[^`]*```', '', text)
        
        return text.strip()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key"""
        engine_id = f"{self.engine.__class__.__name__}_{getattr(self.engine, 'model', 'default')}"
        content = f"{text}_{engine_id}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _cache_result(self, key: str, result: Tuple[str, List[SynthesisUnit]]):
        """Cache result with LRU eviction"""
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest
        self.cache[key] = result
    
    def _llm_process(self, text: str) -> Optional[Tuple[str, List[SynthesisUnit]]]:
        """Process text through two-pass LLM"""
        
        start_time = time.time()
        self.metrics['llm_calls'] += 1
        
        # Pass A: STRUCTURE
        structure_prompt = f"{self.STRUCTURE_PROMPT}\n\nText to convert:\n{text}"
        structure_response = self.engine.generate(
            structure_prompt, 
            temperature=self.structure_temp,
            timeout=self.structure_timeout
        )
        
        if not structure_response:
            logger.warning("Structure pass failed")
            return None
        
        # Parse JSON
        try:
            structured = json.loads(structure_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from structure pass: {e}")
            return None
        
        # Validate and repair
        structured = self.guard.validate_and_repair(structured)
        if not structured:
            logger.warning("Structure validation failed")
            return None
        
        # Pass B: REFINE
        refine_prompt = f"{self.REFINE_PROMPT}\n\nJSON to refine:\n{json.dumps(structured)}"
        refine_response = self.engine.generate(
            refine_prompt,
            temperature=self.refine_temp,
            timeout=self.refine_timeout
        )
        
        if not refine_response:
            logger.warning("Refine pass failed, using structure output")
            refined = structured
        else:
            try:
                refined = json.loads(refine_response)
                
                # Verify same IDs
                if not self._verify_same_ids(structured, refined):
                    logger.warning("Refine pass changed IDs, using structure output")
                    refined = structured
                    
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from refine pass, using structure output")
                refined = structured
        
        # Add frame budgets
        refined = self.budgeter.add_frame_budgets(refined)
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['avg_latency'] = (self.metrics['avg_latency'] + latency) / 2
        
        # Convert to synthesis units
        formatted, units = self._json_to_output(refined)
        
        return (formatted, units)
    
    def _verify_same_ids(self, original: Dict, refined: Dict) -> bool:
        """Verify refine pass kept same IDs"""
        orig_ids = [line['id'] for line in original.get('lines', [])]
        ref_ids = [line['id'] for line in refined.get('lines', [])]
        return orig_ids == ref_ids
    
    def _json_to_output(self, data: Dict) -> Tuple[str, List[SynthesisUnit]]:
        """Convert JSON to formatted text and synthesis units"""
        
        formatted_lines = []
        synthesis_units = []
        speaker_map = {}
        
        for line in data.get('lines', []):
            # Map speaker names to indices
            speaker = line['speaker']
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)
            
            speaker_id = speaker_map[speaker]
            text = line['text']
            
            # Handle pauses
            cues = line.get('cues', {})
            pause_before = cues.get('pause_ms_before', 0)
            pause_after = cues.get('pause_ms_after', 0)
            
            # Add pause before if needed
            if pause_before > 0:
                synthesis_units.append(SynthesisUnit(
                    unit_type='silence',
                    samples=int(pause_before * 24)  # 24 samples per ms at 24kHz
                ))
            
            # Add dialogue unit
            synthesis_units.append(SynthesisUnit(
                unit_type='dialogue',
                speaker_id=speaker_id,
                text=text,
                frames=line.get('line_frames', self.budgeter.calculate_frames(text))
            ))
            
            # Add pause after if needed
            if pause_after > 0:
                synthesis_units.append(SynthesisUnit(
                    unit_type='silence',
                    samples=int(pause_after * 24)
                ))
            
            # Format for text output
            formatted_lines.append(f"Speaker {speaker_id}: {text}")
        
        formatted_text = '\n'.join(formatted_lines)
        return (formatted_text, synthesis_units)
    
    def _regex_fallback(self, text: str) -> Tuple[str, List[SynthesisUnit]]:
        """Fast regex-based processing for clean text"""
        
        lines = text.strip().split('\n')
        formatted_lines = []
        synthesis_units = []
        speaker_map = {}
        
        # Extended speaker pattern
        speaker_pattern = re.compile(r'^\s*([^:：]+)[：:]\s*(.+)', re.UNICODE)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = speaker_pattern.match(line)
            if match:
                speaker = match.group(1).strip()
                content = match.group(2).strip()
                
                # Map speaker to index
                if speaker not in speaker_map:
                    speaker_map[speaker] = len(speaker_map)
                
                speaker_id = speaker_map[speaker]
                
                # Create synthesis unit
                synthesis_units.append(SynthesisUnit(
                    unit_type='dialogue',
                    speaker_id=speaker_id,
                    text=content,
                    frames=self.budgeter.calculate_frames(content)
                ))
                
                formatted_lines.append(f"Speaker {speaker_id}: {content}")
            else:
                # Treat as narration
                synthesis_units.append(SynthesisUnit(
                    unit_type='dialogue',
                    speaker_id=0,
                    text=line,
                    frames=self.budgeter.calculate_frames(line)
                ))
                
                formatted_lines.append(f"Speaker 0: {line}")
        
        formatted_text = '\n'.join(formatted_lines)
        return (formatted_text, synthesis_units)
    
    def get_metrics(self) -> Dict:
        """Get processing metrics"""
        return self.metrics.copy()

# =============================================================================
# Testing
# =============================================================================

def test_llm_processor():
    """Test the LLM text processor"""
    
    print("\n" + "="*60)
    print("Testing LLM Text Processor")
    print("="*60)
    
    processor = LLMTextProcessor({'engine': 'mock'})
    
    # Test 1: Clean text (should use regex path)
    clean_text = """
    Speaker 0: Hello everyone!
    Speaker 1: Hi there!
    """
    
    print("\n[TEST 1] Clean text")
    messiness = processor.scorer.score(clean_text)
    print(f"Messiness score: {messiness:.2f}")
    formatted, units = processor.process(clean_text)
    print(f"Formatted:\n{formatted}")
    print(f"Units: {len(units)}")
    assert messiness < 0.35, "Clean text should have low messiness"
    
    # Test 2: Messy text (should trigger LLM)
    messy_text = """
    Alice: Hello everyone! Welcome to our podcast.
    Bob says "Thanks for having me"
    (pause)
    They continue talking about AI...
    """
    
    print("\n[TEST 2] Messy text")
    messiness = processor.scorer.score(messy_text)
    print(f"Messiness score: {messiness:.2f}")
    formatted, units = processor.process(messy_text, force_llm=True)
    print(f"Formatted:\n{formatted}")
    print(f"Units: {len(units)}")
    
    # Test 3: Frame budgeting
    print("\n[TEST 3] Frame budgeting")
    budgeter = FrameBudgeter()
    text = "Hello world, this is a test."  # 6 words
    frames = budgeter.calculate_frames(text)
    expected_frames = round((6 / 2.75) * 7.5)
    print(f"Text: {text}")
    print(f"Frames: {frames} (expected: {expected_frames})")
    assert frames == expected_frames, "Frame calculation mismatch"
    
    # Test 4: Synthesis units
    print("\n[TEST 4] Synthesis units")
    unit1 = SynthesisUnit(unit_type='dialogue', speaker_id=0, text="Hello", frames=10)
    unit2 = SynthesisUnit(unit_type='silence', samples=2400)
    print(f"Dialogue unit: {unit1.is_dialogue()}")
    print(f"Silence unit: {unit2.is_silence()}")
    assert unit1.is_dialogue() and unit2.is_silence()
    
    print("\n[TEST 5] Metrics")
    print(processor.get_metrics())
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_llm_processor()
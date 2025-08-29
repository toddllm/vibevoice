#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced VibeVoice system
Tests that catch regressions and ensure acceptance criteria
"""

import unittest
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch

# Import components to test
from llm_text_processor import (
    LLMTextProcessor, MessinessScorer, FrameBudgeter,
    SchemaGuard, SynthesisUnit, MockEngine
)
from voice_forge import (
    VoiceForge, VoiceQualityChecker, VariantGenerator
)
try:
    from vibevoice_production_v3 import (
        EnhancedVibeVoiceServer, EnhancedAudioStitcher,
        load_config
    )
except ImportError:
    # For testing without full import chain
    EnhancedVibeVoiceServer = None
    EnhancedAudioStitcher = None
    
    def load_config(path=None):
        return {
            'llm': {'enabled': False},
            'voice_forge': {'base_dir': 'demo/voices'},
            'audio': {'crossfade_ms': 50}
        }

# =============================================================================
# Test 1: Schema Lock - REFINE must preserve IDs
# =============================================================================

class TestSchemaLock(unittest.TestCase):
    """Ensure REFINE pass preserves ID structure"""
    
    def test_refine_preserves_ids(self):
        """REFINE must return same ids length and order"""
        
        processor = LLMTextProcessor({'engine': 'mock'})
        
        original = {
            'version': 'vv-1',
            'language': 'en',
            'lines': [
                {'id': 0, 'speaker': 'Alice', 'type': 'dialogue', 'text': 'Hello'},
                {'id': 1, 'speaker': 'Bob', 'type': 'dialogue', 'text': 'Hi'},
                {'id': 2, 'speaker': 'Alice', 'type': 'dialogue', 'text': 'Bye'}
            ]
        }
        
        # Mock refine that tries to reorder
        refined_bad = {
            'version': 'vv-1',
            'language': 'en',
            'lines': [
                {'id': 1, 'speaker': 'Bob', 'type': 'dialogue', 'text': 'Hi'},
                {'id': 0, 'speaker': 'Alice', 'type': 'dialogue', 'text': 'Hello'},
                {'id': 2, 'speaker': 'Alice', 'type': 'dialogue', 'text': 'Bye'}
            ]
        }
        
        # Verify detection of ID changes
        same_ids = processor._verify_same_ids(original, refined_bad)
        self.assertFalse(same_ids, "Should detect ID reordering")
        
        # Good refine preserves IDs
        refined_good = {
            'version': 'vv-1',
            'language': 'en',
            'lines': [
                {'id': 0, 'speaker': 'Alice', 'type': 'dialogue', 'text': 'Hello!'},
                {'id': 1, 'speaker': 'Bob', 'type': 'dialogue', 'text': 'Hi there!'},
                {'id': 2, 'speaker': 'Alice', 'type': 'dialogue', 'text': 'Goodbye!'}
            ]
        }
        
        same_ids = processor._verify_same_ids(original, refined_good)
        self.assertTrue(same_ids, "Should accept preserved IDs")

# =============================================================================
# Test 2: Budget Accuracy - Frame calculations
# =============================================================================

class TestBudgetAccuracy(unittest.TestCase):
    """Ensure frame budgeting is accurate"""
    
    def test_frame_calculation(self):
        """sum(line_frames) ≈ target_frames ± 1 frame per chunk"""
        
        budgeter = FrameBudgeter()
        
        # Test known word counts
        test_cases = [
            ("Hello world", 2, round((2/2.75)*7.5)),  # 2 words
            ("This is a test sentence", 5, round((5/2.75)*7.5)),  # 5 words
            ("I'm testing contractions here", 4, round((4/2.75)*7.5)),  # 4 words
        ]
        
        for text, expected_words, expected_frames in test_cases:
            frames = budgeter.calculate_frames(text)
            self.assertEqual(frames, expected_frames, 
                           f"Frame calculation wrong for '{text}'")
        
        # Test chunk accuracy
        chunks = [
            "This is the first chunk of text.",  # 7 words
            "Here is another chunk.",  # 4 words
            "Final chunk here."  # 3 words
        ]
        
        total_frames = sum(budgeter.calculate_frames(chunk) for chunk in chunks)
        total_words = 14
        expected_total = round((total_words / 2.75) * 7.5)
        
        # Should be within ±1 frame per chunk
        self.assertLessEqual(abs(total_frames - expected_total), len(chunks),
                           "Total frames drift too much")

# =============================================================================
# Test 3: Pause Fidelity - Exact silence samples
# =============================================================================

class TestPauseFidelity(unittest.TestCase):
    """Ensure pauses produce exact silence samples"""
    
    def test_pause_samples(self):
        """Requested pauses produce exact ms * 24 samples"""
        
        # Test pause units
        pause_100ms = SynthesisUnit('silence', samples=100*24)
        self.assertEqual(pause_100ms.samples, 2400, "100ms should be 2400 samples")
        
        pause_500ms = SynthesisUnit('silence', samples=500*24)
        self.assertEqual(pause_500ms.samples, 12000, "500ms should be 12000 samples")
        
        # Test stitcher generates correct silence
        stitcher = EnhancedAudioStitcher()
        
        units = [
            (SynthesisUnit('dialogue', speaker_id=0, text="Before"), np.ones(1000)),
            (SynthesisUnit('silence', samples=2400), None),
            (SynthesisUnit('dialogue', speaker_id=0, text="After"), np.ones(1000))
        ]
        
        stitched = stitcher.stitch_units(units)
        
        # Should have exact length: 1000 + 2400 + 1000
        expected_length = 4400
        self.assertEqual(len(stitched), expected_length, 
                       f"Stitched length {len(stitched)} != expected {expected_length}")
        
        # Check silence region is actually zeros
        silence_region = stitched[1000:3400]
        self.assertTrue(np.allclose(silence_region, 0), 
                       "Silence region should be zeros")

# =============================================================================
# Test 4: Crossfade Rules
# =============================================================================

class TestCrossfadeRules(unittest.TestCase):
    """Test crossfade application rules"""
    
    def test_no_crossfade_across_speakers(self):
        """Zero crossfade across speaker change"""
        
        stitcher = EnhancedAudioStitcher(crossfade_ms=50)
        
        # Different speakers - no crossfade
        units = [
            (SynthesisUnit('dialogue', speaker_id=0, text="Speaker 0"), np.ones(5000)),
            (SynthesisUnit('dialogue', speaker_id=1, text="Speaker 1"), np.ones(5000))
        ]
        
        stitched = stitcher.stitch_units(units)
        
        # Should be exact concatenation (no crossfade)
        self.assertEqual(len(stitched), 10000, 
                       "Different speakers should not crossfade")
    
    def test_no_crossfade_with_silence(self):
        """Zero crossfade into/from silence"""
        
        stitcher = EnhancedAudioStitcher(crossfade_ms=50)
        
        units = [
            (SynthesisUnit('dialogue', speaker_id=0, text="Before"), np.ones(5000)),
            (SynthesisUnit('silence', samples=1000), None),
            (SynthesisUnit('dialogue', speaker_id=0, text="After"), np.ones(5000))
        ]
        
        stitched = stitcher.stitch_units(units)
        
        # Should be exact: 5000 + 1000 + 5000
        self.assertEqual(len(stitched), 11000,
                       "Silence boundaries should not crossfade")
    
    def test_crossfade_same_speaker(self):
        """Equal-power crossfade within same speaker only"""
        
        stitcher = EnhancedAudioStitcher(crossfade_ms=50)
        crossfade_samples = 50 * 24  # 1200 samples
        
        # Same speaker - should crossfade
        units = [
            (SynthesisUnit('dialogue', speaker_id=0, text="First"), np.ones(5000)),
            (SynthesisUnit('dialogue', speaker_id=0, text="Second"), np.ones(5000))
        ]
        
        stitched = stitcher.stitch_units(units)
        
        # Length should be reduced by crossfade
        expected = 10000 - crossfade_samples
        self.assertEqual(len(stitched), expected,
                       f"Same speaker should crossfade, expected {expected}, got {len(stitched)}")

# =============================================================================
# Test 5: LLM Fallback
# =============================================================================

class TestLLMFallback(unittest.TestCase):
    """Test fallback when LLM fails"""
    
    def test_fallback_on_timeout(self):
        """Pipeline produces valid Speaker N: text via regex on LLM timeout"""
        
        # Create processor with failing engine
        class FailingEngine(MockEngine):
            def generate(self, prompt, temperature, timeout):
                return None  # Simulate timeout
        
        processor = LLMTextProcessor({'engine': 'mock'})
        processor.engine = FailingEngine()
        
        # Process text that would normally trigger LLM
        messy_text = """
        Alice: Hello there!
        Bob: Hi Alice!
        (pause for effect)
        They continue talking...
        """
        
        formatted, units = processor.process(messy_text, force_llm=True)
        
        # Should fallback to regex and produce valid output
        self.assertIn("Speaker 0:", formatted, "Should have Speaker 0")
        self.assertIn("Speaker 1:", formatted, "Should have Speaker 1")
        self.assertGreater(len(units), 0, "Should produce units")
        
        # Check metrics
        self.assertEqual(processor.metrics['fallbacks'], 1, 
                       "Should count fallback")

# =============================================================================
# Test 6: Voice Import Quality Gates
# =============================================================================

class TestVoiceImport(unittest.TestCase):
    """Test voice import rejection with actionable reasons"""
    
    def test_reject_short_audio(self):
        """Reject audio < 15s with clear reason"""
        
        forge = VoiceForge()
        
        # Create 10s audio (too short)
        short_audio = np.sin(2*np.pi*440*np.linspace(0, 10, 240000))
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            import scipy.io.wavfile as wav
            wav.write(tmp.name, 24000, (short_audio * 32767).astype(np.int16))
            
            result = forge.import_voice(tmp.name, {
                'name': 'Test',
                'consent': True
            })
            
            self.assertFalse(result['success'], "Should reject short audio")
            self.assertTrue(any('15' in str(issue) for issue in result['issues']),
                          "Should mention 15s minimum")
    
    def test_reject_clipped_audio(self):
        """Reject heavily clipped audio"""
        
        checker = VoiceQualityChecker()
        
        # Create clipped audio
        audio = np.ones(48000) * 1.5  # Will clip
        audio = np.clip(audio, -1, 1)
        
        quality = checker.analyze(audio, 24000)
        
        self.assertGreater(quality.clipping_ratio, 0.001, 
                         "Should detect clipping")
        self.assertFalse(quality.passed, "Should fail quality check")
    
    def test_reject_celebrity_names(self):
        """Refuse celebrity/impersonation names"""
        
        forge = VoiceForge()
        
        # Try celebrity name
        with self.assertRaises(ValueError) as ctx:
            forge.import_voice("dummy.wav", {
                'name': 'Obama',
                'consent': True
            })
        
        self.assertIn("celebrity", str(ctx.exception).lower())

# =============================================================================
# Test 7: Variant Bounds
# =============================================================================

class TestVariantBounds(unittest.TestCase):
    """Test voice variants stay within safe bounds"""
    
    def test_pitch_bounds(self):
        """Pitch shifts never exceed ±2 semitones"""
        
        gen = VariantGenerator()
        
        # Check all recipes
        for recipe_name, recipe in gen.RECIPES.items():
            for op in recipe['ops']:
                if op['type'] == 'pitch':
                    semitones = op['semitones']
                    self.assertLessEqual(abs(semitones), 2,
                                       f"Recipe {recipe_name} pitch exceeds ±2 semitones")
    
    def test_rate_bounds(self):
        """Rate changes stay within 0.95-1.05×"""
        
        gen = VariantGenerator()
        
        for recipe_name, recipe in gen.RECIPES.items():
            for op in recipe['ops']:
                if op['type'] == 'rate':
                    factor = op['factor']
                    self.assertGreaterEqual(factor, 0.95,
                                          f"Recipe {recipe_name} rate too slow")
                    self.assertLessEqual(factor, 1.05,
                                       f"Recipe {recipe_name} rate too fast")
    
    def test_eq_bounds(self):
        """EQ gains stay within ±3dB"""
        
        gen = VariantGenerator()
        
        # Test that EQ is clamped in generation
        audio = np.random.randn(24000)
        
        # Try extreme EQ (should be clamped)
        extreme_recipe = {
            'name': 'Test',
            'ops': [{'type': 'eq_shelf', 'freq': 5000, 'gain_db': 10}]
        }
        
        gen.RECIPES['test_extreme'] = extreme_recipe
        result_audio, metadata = gen.generate_variant(audio, 24000, 'test_extreme')
        
        # Audio should not be blown out
        self.assertLessEqual(np.abs(result_audio).max(), 1.0,
                           "EQ should not cause clipping")

# =============================================================================
# Test 8: Messiness Scoring
# =============================================================================

class TestMessinessScoring(unittest.TestCase):
    """Test messiness scorer accuracy"""
    
    def test_clean_text_low_score(self):
        """Clean Speaker N: format gets < 0.35 score"""
        
        scorer = MessinessScorer()
        
        clean_texts = [
            "Speaker 0: Hello world.\nSpeaker 1: Hi there.",
            "Speaker 0: This is clean.\nSpeaker 0: Very clean.",
            "Speaker 1: Perfect format here."
        ]
        
        for text in clean_texts:
            score = scorer.score(text)
            self.assertLess(score, 0.35,
                          f"Clean text scored {score}, should be < 0.35")
    
    def test_messy_text_high_score(self):
        """Messy text gets >= 0.35 score"""
        
        scorer = MessinessScorer()
        
        messy_texts = [
            "Alice: Hello\nBob says hi\n(pause)",
            "- Point one\n- Point two\n- Point three",
            "This is a paragraph without any speaker tags at all.",
            "Speaker 0: Hi\nAlice: Hello\nMixed formats!"
        ]
        
        for text in messy_texts:
            score = scorer.score(text)
            self.assertGreaterEqual(score, 0.35,
                                  f"Messy text scored {score}, should be >= 0.35")

# =============================================================================
# Acceptance Criteria Tests
# =============================================================================

class TestAcceptanceCriteria(unittest.TestCase):
    """Test MVP acceptance criteria"""
    
    def test_clean_input_no_llm(self):
        """Clean input runs entirely on heuristic path"""
        
        processor = LLMTextProcessor({'engine': 'mock'})
        
        # Track LLM calls
        original_generate = processor.engine.generate
        processor.engine.generate = Mock(side_effect=original_generate)
        
        clean = "Speaker 0: Hello.\nSpeaker 1: Hi."
        formatted, units = processor.process(clean)
        
        # Should not call LLM
        processor.engine.generate.assert_not_called()
        
        # Should produce valid output
        self.assertIn("Speaker 0:", formatted)
        self.assertIn("Speaker 1:", formatted)
    
    def test_messy_converts_correctly(self):
        """Messy paragraph converts with ≤ 2s overhead"""
        
        import time
        
        processor = LLMTextProcessor({'engine': 'mock'})
        
        messy = """
        Alice starts by saying "Welcome everyone to our show!"
        Bob responds with enthusiasm: Thanks for having me, this is great.
        They discuss AI for a while.
        """
        
        start = time.time()
        formatted, units = processor.process(messy, force_llm=True)
        elapsed = time.time() - start
        
        # Should complete quickly (mock engine)
        self.assertLess(elapsed, 2.0, f"Processing took {elapsed}s, should be < 2s")
        
        # Should have speaker format
        self.assertIn("Speaker", formatted)
        
        # Should have multiple units
        self.assertGreater(len(units), 1)
    
    def test_pause_silence_no_crossfade(self):
        """Pauses show as silence units with no crossfade"""
        
        # Create units with pause
        processor = LLMTextProcessor({'engine': 'mock'})
        
        # Mock response with pause
        mock_json = {
            'version': 'vv-1',
            'lines': [
                {'id': 0, 'speaker': 'Alice', 'type': 'dialogue', 
                 'text': 'Hello', 'cues': {'pause_ms_after': 200}},
                {'id': 1, 'speaker': 'Bob', 'type': 'dialogue', 'text': 'Hi'}
            ]
        }
        
        formatted, units = processor._json_to_output(mock_json)
        
        # Should have silence unit
        silence_units = [u for u in units if u.is_silence()]
        self.assertEqual(len(silence_units), 1, "Should have one pause")
        self.assertEqual(silence_units[0].samples, 200*24, "Pause should be 200ms")
    
    def test_voice_import_success(self):
        """45s recording yields registered voice + variants"""
        
        forge = VoiceForge()
        
        # Create 45s audio
        audio = np.sin(2*np.pi*440*np.linspace(0, 45, 45*24000))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import scipy.io.wavfile as wav
            wav.write(tmp.name, 24000, (audio * 32767).astype(np.int16))
            
            try:
                result = forge.import_voice(tmp.name, {
                    'name': 'TestVoice',
                    'lang': 'en',
                    'gender': 'neutral',
                    'consent': True
                }, auto_variants=True)
                
                self.assertTrue(result['success'], "Import should succeed")
                self.assertIn('voice_id', result)
                self.assertGreater(len(result.get('variants', [])), 0,
                              "Should create variants")
                
            finally:
                os.unlink(tmp.name)
                # Clean up created files
                if 'voice_path' in result:
                    if os.path.exists(result['voice_path']):
                        os.unlink(result['voice_path'])
                    yaml_path = f"{result['voice_path']}.yaml"
                    if os.path.exists(yaml_path):
                        os.unlink(yaml_path)

# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all test suites"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSchemaLock,
        TestBudgetAccuracy,
        TestPauseFidelity,
        TestCrossfadeRules,
        TestLLMFallback,
        TestVoiceImport,
        TestVariantBounds,
        TestMessinessScoring,
        TestAcceptanceCriteria
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    if result.wasSuccessful():
        print("\n✅ All regression tests passed!")
    else:
        print("\n❌ Some tests failed - review output above")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
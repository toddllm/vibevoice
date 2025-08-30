#!/usr/bin/env python3
"""
Comprehensive LLM Test Suite for VibeVoice
Test various LLMs against standardized text format conversion cases
"""

import json
import time
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TestCase:
    """Single test case for LLM evaluation"""
    name: str
    category: str
    input_text: str
    expected_speakers: List[int]
    expected_dialogue_count: int
    expected_narrative_count: int
    description: str

@dataclass
class TestResult:
    """Result of running a test case"""
    test_name: str
    success: bool
    actual_output: str
    actual_speakers: List[int] 
    actual_dialogue_count: int
    actual_narrative_count: int
    processing_time: float
    error_message: Optional[str] = None

class LLMTestSuite:
    """Test suite for evaluating LLM text processing performance"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering all format types"""
        return [
            # === BASIC DIALOGUE TESTS ===
            TestCase(
                name="simple_dialogue",
                category="dialogue",
                input_text='Alice: "Hello!" Bob: "Hi there!"',
                expected_speakers=[1, 2],
                expected_dialogue_count=2,
                expected_narrative_count=0,
                description="Basic two-speaker dialogue"
            ),
            
            TestCase(
                name="dialogue_with_narrative",
                category="dialogue", 
                input_text='"Welcome," said Alice. Bob nodded. "Thanks," he replied.',
                expected_speakers=[0, 1, 2],
                expected_dialogue_count=2,
                expected_narrative_count=1,
                description="Mixed dialogue and narrative"
            ),
            
            # === NARRATIVE TESTS ===
            TestCase(
                name="pure_narrative",
                category="narrative",
                input_text="Alice welcomed everyone to the meeting. Bob thanked her for the opportunity. The audience listened intently.",
                expected_speakers=[0],
                expected_dialogue_count=0,
                expected_narrative_count=3,
                description="Pure narrative with no dialogue"
            ),
            
            TestCase(
                name="narrative_with_embedded_dialogue",
                category="narrative",
                input_text='Alice walked in. "This is exciting," she mentioned. Bob agreed and started his presentation.',
                expected_speakers=[0, 1],
                expected_dialogue_count=1,
                expected_narrative_count=2,
                description="Narrative with embedded quoted dialogue"
            ),
            
            # === SCRIPT TESTS ===
            TestCase(
                name="screenplay_format",
                category="script",
                input_text='INT. COFFEE SHOP - DAY\n\n- Alice enters the bustling coffee shop\n- Bob waves from a corner table\n\nBob says "Over here!"\nAlice responds with "Sorry I\'m late"',
                expected_speakers=[0, 1, 2],
                expected_dialogue_count=2,
                expected_narrative_count=1,
                description="Screenplay format with stage directions"
            ),
            
            TestCase(
                name="action_dialogue_mix",
                category="script",
                input_text='- Character walks in\n"Welcome to the show!" the host announces.\n- Audience applauds\n"Thank you," the guest replies.',
                expected_speakers=[0, 1, 2],
                expected_dialogue_count=2,
                expected_narrative_count=2,
                description="Action lines mixed with dialogue"
            ),
            
            # === MIXED FORMAT TESTS ===
            TestCase(
                name="format_chaos",
                category="mixed",
                input_text='Speaker 1: Let\'s begin\n• Alice: "I\'ll go first"\n- Bob: Great idea!\n"Count me in," Charlie says\n[Everyone laughs]',
                expected_speakers=[0, 1, 2, 3],
                expected_dialogue_count=3,
                expected_narrative_count=2,
                description="Multiple formatting styles in one text"
            ),
            
            TestCase(
                name="inconsistent_speaker_tags",
                category="mixed",
                input_text='Alice: Welcome!\nSpeaker 2: Thanks\n• Bob says "Great!"\nCharlie: "Awesome"',
                expected_speakers=[1, 2, 3, 4],
                expected_dialogue_count=4,
                expected_narrative_count=0,
                description="Inconsistent speaker tagging formats"
            ),
            
            # === COMPLEX TESTS ===
            TestCase(
                name="multi_paragraph_story",
                category="complex",
                input_text='"Good morning, everyone," Alice began. "Thank you all for coming."\n\nBob looked up from his laptop. "No problem at all. This project is important."\n\nThe room fell silent as Alice continued. "We have twelve weeks to complete this."',
                expected_speakers=[0, 1, 2],
                expected_dialogue_count=4,
                expected_narrative_count=2,
                description="Multi-paragraph story with mixed dialogue"
            ),
            
            TestCase(
                name="nested_quotes_and_attributions",
                category="complex",
                input_text='Alice said, "Bob told me, \'This project is crucial.\'" Bob replied, "I never said that exactly."',
                expected_speakers=[0, 1, 2],
                expected_dialogue_count=2,
                expected_narrative_count=1,
                description="Nested quotes with speech attributions"
            ),
            
            # === EDGE CASES ===
            TestCase(
                name="no_speakers",
                category="edge",
                input_text="The meeting room was empty. Sunlight streamed through the windows. Papers rustled in the breeze.",
                expected_speakers=[0],
                expected_dialogue_count=0,
                expected_narrative_count=3,
                description="Pure description with no speakers"
            ),
            
            TestCase(
                name="single_word_responses",
                category="edge", 
                input_text='Alice: "Yes." Bob: "No." Charlie: "Maybe."',
                expected_speakers=[1, 2, 3],
                expected_dialogue_count=3,
                expected_narrative_count=0,
                description="Very short dialogue responses"
            ),
            
            TestCase(
                name="punctuation_heavy",
                category="edge",
                input_text='"Wait... what?!" Alice exclaimed. "I don\'t understand," Bob said nervously. "Calm down," Charlie intervened.',
                expected_speakers=[0, 1, 2, 3],
                expected_dialogue_count=3,
                expected_narrative_count=1,
                description="Heavy punctuation and emotional dialogue"
            )
        ]
    
    def run_test_case(self, test_case: TestCase) -> TestResult:
        """Run a single test case against the LLM"""
        print(f"\n🧪 Testing: {test_case.name}")
        print(f"   Category: {test_case.category}")
        print(f"   Input: {test_case.input_text[:100]}...")
        
        try:
            start_time = time.time()
            
            # Call normalize endpoint
            response = requests.post(
                f"{self.base_url}/normalize",
                json={"text": test_case.input_text},
                timeout=180  # 3 minutes max
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                return TestResult(
                    test_name=test_case.name,
                    success=False,
                    actual_output="",
                    actual_speakers=[],
                    actual_dialogue_count=0,
                    actual_narrative_count=0,
                    processing_time=processing_time,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
            
            result = response.json()
            actual_output = result.get('normalized', '')
            
            # Parse actual results
            actual_speakers = self._extract_speakers(actual_output)
            actual_dialogue_count, actual_narrative_count = self._count_dialogue_vs_narrative(actual_output)
            
            # Determine success
            success = self._evaluate_success(test_case, actual_speakers, actual_dialogue_count, actual_narrative_count)
            
            print(f"   ✓ Completed in {processing_time:.1f}s")
            print(f"   Speakers: {actual_speakers} (expected: {test_case.expected_speakers})")
            
            return TestResult(
                test_name=test_case.name,
                success=success,
                actual_output=actual_output,
                actual_speakers=actual_speakers,
                actual_dialogue_count=actual_dialogue_count,
                actual_narrative_count=actual_narrative_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                success=False,
                actual_output="",
                actual_speakers=[],
                actual_dialogue_count=0,
                actual_narrative_count=0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _extract_speakers(self, output: str) -> List[int]:
        """Extract unique speaker IDs from output"""
        import re
        speakers = set()
        for line in output.split('\n'):
            match = re.match(r'Speaker (\d+):', line.strip())
            if match:
                speakers.add(int(match.group(1)))
        return sorted(list(speakers))
    
    def _count_dialogue_vs_narrative(self, output: str) -> Tuple[int, int]:
        """Count dialogue vs narrative lines"""
        dialogue_count = 0
        narrative_count = 0
        
        for line in output.split('\n'):
            if line.strip().startswith('Speaker 0:'):
                narrative_count += 1
            elif line.strip().startswith('Speaker '):
                dialogue_count += 1
                
        return dialogue_count, narrative_count
    
    def _evaluate_success(self, test_case: TestCase, actual_speakers: List[int], 
                         actual_dialogue: int, actual_narrative: int) -> bool:
        """Evaluate if test case passed"""
        # Check speaker assignment
        expected_speakers = set(test_case.expected_speakers)
        actual_speakers_set = set(actual_speakers)
        
        # Allow some flexibility in speaker assignment
        if len(actual_speakers_set) != len(expected_speakers):
            return False
            
        # Check dialogue/narrative split (allow ±1 tolerance)
        dialogue_ok = abs(actual_dialogue - test_case.expected_dialogue_count) <= 1
        narrative_ok = abs(actual_narrative - test_case.expected_narrative_count) <= 1
        
        return dialogue_ok and narrative_ok
    
    def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        print("🚀 Running LLM Test Suite for VibeVoice")
        print("=" * 50)
        
        results = []
        start_time = time.time()
        
        for test_case in self.test_cases:
            result = self.run_test_case(test_case)
            results.append(result)
            
            if result.success:
                print(f"   ✅ PASS")
            else:
                print(f"   ❌ FAIL: {result.error_message or 'Output mismatch'}")
        
        # Generate summary
        total_time = time.time() - start_time
        passed = len([r for r in results if r.success])
        failed = len(results) - passed
        
        summary = {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(results) * 100,
            "total_time": total_time,
            "avg_time_per_test": total_time / len(results),
            "results": results
        }
        
        print("\n📊 TEST SUMMARY")
        print("=" * 30)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Time: {summary['total_time']:.1f}s")
        print(f"Avg Time/Test: {summary['avg_time_per_test']:.1f}s")
        
        return summary
    
    def run_category(self, category: str) -> Dict:
        """Run tests for a specific category"""
        category_tests = [tc for tc in self.test_cases if tc.category == category]
        print(f"🧪 Running {category.upper()} tests ({len(category_tests)} cases)")
        
        results = []
        for test_case in category_tests:
            result = self.run_test_case(test_case)
            results.append(result)
        
        passed = len([r for r in results if r.success])
        return {
            "category": category,
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"llm_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
        return filename

def main():
    """Run the test suite"""
    import sys
    
    suite = LLMTestSuite()
    
    if len(sys.argv) > 1:
        # Run specific category
        category = sys.argv[1]
        results = suite.run_category(category)
    else:
        # Run all tests
        results = suite.run_all_tests()
    
    # Save results
    suite.save_results(results)
    
    return results['success_rate'] > 80 if 'success_rate' in results else False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
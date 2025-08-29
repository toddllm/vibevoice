#!/usr/bin/env python3
"""
VibeVoice Voice Manager
Comprehensive voice management for all built-in voices
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.io.wavfile as wavfile

class VoiceProfile:
    """Detailed voice profile with metadata"""
    
    def __init__(self, id: str, path: str):
        self.id = id
        self.path = path
        self.name = id.replace('.wav', '')
        
        # Parse voice metadata from filename
        parts = self.name.split('-')
        if len(parts) >= 2:
            self.language = parts[0]  # en, zh, in
            name_parts = parts[1].split('_')
            self.speaker_name = name_parts[0]
            self.gender = name_parts[1] if len(name_parts) > 1 else 'unknown'
            self.has_bgm = 'bgm' in self.name.lower()
        else:
            self.language = 'en'
            self.speaker_name = self.name
            self.gender = 'unknown'
            self.has_bgm = False
        
        # Load audio properties
        self._load_properties()
    
    def _load_properties(self):
        """Load audio properties"""
        try:
            sr, wav = wavfile.read(self.path)
            self.sample_rate = sr
            self.duration = len(wav) / sr
            self.samples = len(wav)
        except:
            self.sample_rate = 16000
            self.duration = 0
            self.samples = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'path': self.path,
            'name': self.name,
            'speaker_name': self.speaker_name,
            'language': self.language,
            'gender': self.gender,
            'has_bgm': self.has_bgm,
            'sample_rate': self.sample_rate,
            'duration': round(self.duration, 2),
            'display_name': self.get_display_name()
        }
    
    def get_display_name(self) -> str:
        """Get user-friendly display name"""
        display = self.speaker_name
        
        # Add language indicator
        lang_map = {'en': '🇬🇧', 'zh': '🇨🇳', 'in': '🇮🇳'}
        if self.language in lang_map:
            display = f"{lang_map[self.language]} {display}"
        
        # Add gender indicator
        if self.gender == 'woman':
            display += " (Female)"
        elif self.gender == 'man':
            display += " (Male)"
        
        # Add BGM indicator
        if self.has_bgm:
            display += " 🎵"
        
        return display


class VoiceManager:
    """Comprehensive voice management system"""
    
    # Built-in voices from VibeVoice
    BUILT_IN_VOICES = {
        # English voices
        'en-Alice_woman': {
            'description': 'Female English speaker, clear and professional',
            'best_for': 'Podcasts, narration, general use',
            'characteristics': 'Clear, friendly, versatile'
        },
        'en-Carter_man': {
            'description': 'Male English speaker, deep and authoritative',
            'best_for': 'News, documentaries, formal content',
            'characteristics': 'Deep, professional, authoritative'
        },
        'en-Frank_man': {
            'description': 'Male English speaker, casual and friendly',
            'best_for': 'Casual conversations, tutorials',
            'characteristics': 'Warm, approachable, conversational'
        },
        'en-Mary_woman_bgm': {
            'description': 'Female English speaker with background music',
            'best_for': 'Storytelling, creative content',
            'characteristics': 'Expressive, musical, atmospheric'
        },
        'en-Maya_woman': {
            'description': 'Female English speaker, youthful and energetic',
            'best_for': 'Educational content, presentations',
            'characteristics': 'Bright, clear, enthusiastic'
        },
        
        # International voice
        'in-Samuel_man': {
            'description': 'Male Indian English speaker',
            'best_for': 'International content, tech tutorials',
            'characteristics': 'Clear accent, professional'
        },
        
        # Chinese voices
        'zh-Anchen_man_bgm': {
            'description': 'Male Chinese speaker with background music',
            'best_for': 'Chinese podcasts, storytelling',
            'characteristics': 'Warm, atmospheric, engaging'
        },
        'zh-Bowen_man': {
            'description': 'Male Chinese speaker, professional',
            'best_for': 'Chinese news, formal content',
            'characteristics': 'Clear, authoritative, formal'
        },
        'zh-Xinran_woman': {
            'description': 'Female Chinese speaker, pleasant and clear',
            'best_for': 'Chinese narration, educational content',
            'characteristics': 'Gentle, clear, professional'
        }
    }
    
    # Speaker assignment recommendations for multi-speaker scenarios
    MULTI_SPEAKER_RECOMMENDATIONS = {
        'podcast': {
            'host': ['en-Alice_woman', 'en-Maya_woman'],
            'guest': ['en-Carter_man', 'en-Frank_man'],
            'narrator': ['en-Mary_woman_bgm']
        },
        'interview': {
            'interviewer': ['en-Alice_woman', 'en-Carter_man'],
            'interviewee': ['en-Frank_man', 'en-Maya_woman']
        },
        'story': {
            'narrator': ['en-Mary_woman_bgm', 'zh-Anchen_man_bgm'],
            'character1': ['en-Alice_woman', 'zh-Xinran_woman'],
            'character2': ['en-Frank_man', 'zh-Bowen_man']
        },
        'meeting': {
            'manager': ['en-Carter_man', 'zh-Bowen_man'],
            'team_member1': ['en-Alice_woman', 'zh-Xinran_woman'],
            'team_member2': ['en-Frank_man', 'in-Samuel_man']
        }
    }
    
    def __init__(self, voices_dir: str = "demo/voices"):
        self.voices_dir = voices_dir
        self.voices: Dict[str, VoiceProfile] = {}
        self.load_voices()
    
    def load_voices(self):
        """Load all available voices"""
        if not os.path.exists(self.voices_dir):
            print(f"Warning: Voices directory not found at {self.voices_dir}")
            return
        
        # Load all WAV files
        for filename in os.listdir(self.voices_dir):
            if filename.endswith('.wav'):
                path = os.path.join(self.voices_dir, filename)
                voice_id = filename.replace('.wav', '')
                self.voices[voice_id] = VoiceProfile(filename, path)
        
        print(f"Loaded {len(self.voices)} voices")
    
    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get a specific voice profile"""
        # Try exact match first
        if voice_id in self.voices:
            return self.voices[voice_id]
        
        # Try without extension
        voice_id_no_ext = voice_id.replace('.wav', '')
        if voice_id_no_ext in self.voices:
            return self.voices[voice_id_no_ext]
        
        # Try partial match
        for vid, voice in self.voices.items():
            if voice_id.lower() in vid.lower() or vid.lower() in voice_id.lower():
                return voice
        
        return None
    
    def get_voices_by_language(self, language: str) -> List[VoiceProfile]:
        """Get all voices for a specific language"""
        return [v for v in self.voices.values() if v.language == language]
    
    def get_voices_by_gender(self, gender: str) -> List[VoiceProfile]:
        """Get all voices for a specific gender"""
        return [v for v in self.voices.values() if v.gender == gender]
    
    def get_voices_with_bgm(self) -> List[VoiceProfile]:
        """Get all voices with background music"""
        return [v for v in self.voices.values() if v.has_bgm]
    
    def get_voice_recommendations(self, num_speakers: int, scenario: str = 'podcast') -> List[str]:
        """Get voice recommendations for multi-speaker scenarios"""
        recommendations = []
        
        if scenario in self.MULTI_SPEAKER_RECOMMENDATIONS:
            roles = self.MULTI_SPEAKER_RECOMMENDATIONS[scenario]
            for role, voices in roles.items():
                for voice in voices:
                    if voice in self.voices and len(recommendations) < num_speakers:
                        recommendations.append(voice)
        
        # Fill with defaults if needed
        all_voices = list(self.voices.keys())
        while len(recommendations) < num_speakers and len(recommendations) < len(all_voices):
            for voice_id in all_voices:
                if voice_id not in recommendations:
                    recommendations.append(voice_id)
                    break
        
        return recommendations[:num_speakers]
    
    def assign_voices_to_speakers(self, speakers: List[str], scenario: str = 'podcast') -> Dict[str, str]:
        """Automatically assign voices to speakers based on scenario"""
        assignments = {}
        recommendations = self.get_voice_recommendations(len(speakers), scenario)
        
        for i, speaker in enumerate(speakers):
            if i < len(recommendations):
                assignments[speaker] = recommendations[i]
            else:
                # Cycle through available voices
                assignments[speaker] = recommendations[i % len(recommendations)]
        
        return assignments
    
    def get_voice_info(self) -> Dict:
        """Get comprehensive voice information"""
        return {
            'total_voices': len(self.voices),
            'languages': list(set(v.language for v in self.voices.values())),
            'voices': [v.to_dict() for v in self.voices.values()],
            'by_language': {
                'english': len(self.get_voices_by_language('en')),
                'chinese': len(self.get_voices_by_language('zh')),
                'international': len(self.get_voices_by_language('in'))
            },
            'by_gender': {
                'male': len(self.get_voices_by_gender('man')),
                'female': len(self.get_voices_by_gender('woman'))
            },
            'with_bgm': len(self.get_voices_with_bgm())
        }
    
    def export_voice_catalog(self, output_path: str = "voice_catalog.json"):
        """Export voice catalog to JSON"""
        catalog = {
            'version': '1.0',
            'total_voices': len(self.voices),
            'voices': {},
            'recommendations': self.MULTI_SPEAKER_RECOMMENDATIONS
        }
        
        for voice_id, voice in self.voices.items():
            catalog['voices'][voice_id] = {
                **voice.to_dict(),
                'description': self.BUILT_IN_VOICES.get(voice_id, {}).get('description', ''),
                'best_for': self.BUILT_IN_VOICES.get(voice_id, {}).get('best_for', ''),
                'characteristics': self.BUILT_IN_VOICES.get(voice_id, {}).get('characteristics', '')
            }
        
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        print(f"Voice catalog exported to {output_path}")
        return catalog


def test_voice_manager():
    """Test the voice manager"""
    print("\n" + "="*60)
    print(" VibeVoice Voice Manager Test")
    print("="*60)
    
    vm = VoiceManager()
    
    # Display voice information
    info = vm.get_voice_info()
    print(f"\nTotal voices: {info['total_voices']}")
    print(f"Languages: {', '.join(info['languages'])}")
    print(f"English voices: {info['by_language']['english']}")
    print(f"Chinese voices: {info['by_language']['chinese']}")
    print(f"Male voices: {info['by_gender']['male']}")
    print(f"Female voices: {info['by_gender']['female']}")
    print(f"Voices with BGM: {info['with_bgm']}")
    
    # Display all voices
    print("\n📢 Available Voices:")
    for voice in vm.voices.values():
        print(f"  {voice.get_display_name()}")
        print(f"    ID: {voice.id}")
        print(f"    Duration: {voice.duration:.1f}s @ {voice.sample_rate}Hz")
        if voice.id in vm.BUILT_IN_VOICES:
            print(f"    Description: {vm.BUILT_IN_VOICES[voice.id]['description']}")
        print()
    
    # Test multi-speaker assignment
    print("\n🎭 Multi-Speaker Recommendations:")
    test_speakers = ['Alice', 'Bob', 'Charlie']
    
    for scenario in ['podcast', 'interview', 'story', 'meeting']:
        assignments = vm.assign_voices_to_speakers(test_speakers, scenario)
        print(f"\n  {scenario.title()} scenario:")
        for speaker, voice_id in assignments.items():
            voice = vm.get_voice(voice_id)
            if voice:
                print(f"    {speaker} → {voice.get_display_name()}")
    
    # Export catalog
    catalog = vm.export_voice_catalog()
    print(f"\n✅ Voice catalog exported with {len(catalog['voices'])} voices")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_voice_manager()
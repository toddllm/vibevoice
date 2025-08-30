#!/usr/bin/env python3
"""
Voice Mapping System for Character Assignment
Intelligently assigns voices to characters based on name and gender cues
"""

import os
import glob
import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VoiceMapper:
    """Maps characters to appropriate voices based on available voice library"""
    
    def __init__(self, voice_dir: str = "demo/voices"):
        self.voice_dir = voice_dir
        self.voices = self._scan_voices()
        self.male_voices = [v for v in self.voices if v['gender'] == 'male']
        self.female_voices = [v for v in self.voices if v['gender'] == 'female']
        
        logger.info(f"VoiceMapper initialized: {len(self.male_voices)} male, {len(self.female_voices)} female voices")
    
    def _scan_voices(self) -> List[Dict]:
        """Scan voice directory and categorize voices"""
        voices = []
        
        # Scan main voices directory
        voice_patterns = [
            os.path.join(self.voice_dir, "*.wav"),
            os.path.join(self.voice_dir, "variants", "*.wav")
        ]
        
        for pattern in voice_patterns:
            for voice_file in glob.glob(pattern):
                voice_info = self._parse_voice_filename(voice_file)
                if voice_info:
                    voices.append(voice_info)
        
        return voices
    
    def _parse_voice_filename(self, filepath: str) -> Optional[Dict]:
        """Parse voice filename to extract name, gender, and style"""
        filename = os.path.basename(filepath)
        
        # Pattern: en-Name_gender[_style].wav
        match = re.match(r'(\w+)-(\w+)_(man|woman)(?:_(.+))?\.wav', filename)
        if not match:
            return None
        
        language, name, gender_word, style = match.groups()
        gender = 'male' if gender_word == 'man' else 'female'
        style = style or 'neutral'
        
        return {
            'name': name,
            'gender': gender,
            'language': language,
            'style': style,
            'path': filepath,
            'filename': filename
        }
    
    def get_available_voices_summary(self) -> str:
        """Get formatted summary of available voices for LLM prompt"""
        male_names = [v['name'] for v in self.male_voices[:5]]  # Limit to avoid long prompts
        female_names = [v['name'] for v in self.female_voices[:5]]
        
        return f"""AVAILABLE VOICES:
Male voices: {', '.join(male_names)}
Female voices: {', '.join(female_names)}"""
    
    def assign_voices_to_speakers(self, character_assignments: Dict[int, str]) -> Dict[int, str]:
        """
        Assign appropriate voice files to speaker IDs based on character names
        
        Args:
            character_assignments: {speaker_id: character_name}
            
        Returns:
            {speaker_id: voice_filename}
        """
        assignments = {}
        used_voices = set()
        
        for speaker_id, character_name in character_assignments.items():
            if speaker_id == 0:  # Narrator - use default
                assignments[0] = "demo/voices/en-Alice_woman.wav"
                continue
            
            # Determine likely gender from name
            likely_gender = self._guess_gender_from_name(character_name)
            
            # Get appropriate voice pool
            voice_pool = self.male_voices if likely_gender == 'male' else self.female_voices
            
            # Find unused voice
            for voice in voice_pool:
                if voice['path'] not in used_voices:
                    assignments[speaker_id] = voice['path']
                    used_voices.add(voice['path'])
                    break
            else:
                # Fallback if all voices used
                if voice_pool:
                    assignments[speaker_id] = voice_pool[0]['path']
        
        return assignments
    
    def _guess_gender_from_name(self, name: str) -> str:
        """Guess gender from character name using common patterns"""
        name_lower = name.lower()
        
        # Common male names/patterns
        male_patterns = ['bob', 'frank', 'carter', 'john', 'mike', 'dave', 'tom', 'bill', 'sam', 'alex']
        female_patterns = ['alice', 'emily', 'mary', 'sarah', 'lisa', 'anna', 'maya', 'emma', 'jane']
        
        if any(pattern in name_lower for pattern in male_patterns):
            return 'male'
        elif any(pattern in name_lower for pattern in female_patterns):
            return 'female'
        
        # Default fallback
        return 'male' if len(name) % 2 == 0 else 'female'
    
    def get_voice_assignment_prompt_addition(self) -> str:
        """Get additional prompt text for voice assignment"""
        return f"""
{self.get_available_voices_summary()}

VOICE ASSIGNMENT RULES:
- Speaker 0 (Narrator): Always use Alice (female default)
- Male characters (Bob, Frank, etc.): Assign to male voices (Frank, Carter)  
- Female characters (Alice, Emily, etc.): Assign to female voices (Emily, Maya)
- Include voice assignment in output format: "Speaker N (VoiceName): text"
"""

if __name__ == "__main__":
    # Test the mapper
    mapper = VoiceMapper()
    print("Available voices:")
    for voice in mapper.voices:
        print(f"  {voice['name']} ({voice['gender']}) - {voice['style']}")
    
    # Test assignment
    characters = {0: "Narrator", 1: "Bob", 2: "Alice", 3: "Emily"}
    assignments = mapper.assign_voices_to_speakers(characters)
    print("\nVoice assignments:")
    for speaker_id, voice_path in assignments.items():
        char_name = characters[speaker_id]
        print(f"  Speaker {speaker_id} ({char_name}): {voice_path}")
#!/usr/bin/env python3
"""
Shared Prompt Templates for VibeVoice
Ensures consistent prompting across all interfaces
"""

class VibeVoicePrompts:
    """Centralized prompt templates for text-to-speaker conversion"""
    
    CURRENT_PROMPT = """Convert text to VibeVoice format with speaker assignments and voice mapping.

NARRATOR INSTRUCTIONS:
- Speaker 0 = Narrator (Alice female voice) - handles ALL non-dialogue text
- Split narrative into separate sentences for natural pacing
- Include actions, descriptions, "he said", "she replied", stage directions

DIALOGUE INSTRUCTIONS:
- Extract ONLY quoted text "..." as dialogue
- Assign different speakers (1, 2, 3...) to different characters
- Match character gender to appropriate voices

VOICE ASSIGNMENTS:
- Speaker 0: Alice (narrator, female)
- Male characters (Bob, Frank, etc.) → Frank, Carter, Samuel (male voices)
- Female characters (Alice, Emily, etc.) → Emily, Maya, Mary (female voices)

OUTPUT FORMAT:
1. VibeVoice format (Speaker lines)
2. JSON voice mapping

EXAMPLES:

Input: Alice entered the room. "Hello everyone," she said. Bob waved. "Thanks for coming," he replied.
Output:
Speaker 0: Alice entered the room.
Speaker 1: Hello everyone
Speaker 0: she said. Bob waved.
Speaker 2: Thanks for coming
Speaker 0: he replied.

Voice Mapping:
{{"voice_assignments": {{"0": "Alice", "1": "Emily", "2": "Frank"}}}}

Input: The meeting started. "Welcome," said Frank. Emily nodded approvingly.
Output:
Speaker 0: The meeting started.
Speaker 1: Welcome
Speaker 0: said Frank. Emily nodded approvingly.

Voice Mapping:
{{"voice_assignments": {{"0": "Alice", "1": "Frank"}}}}

---
TEXT: {text}
---
YOUR OUTPUT (VibeVoice format + JSON voice mapping):"""

    @classmethod
    def get_current_prompt(cls, text: str) -> str:
        """Get the current production prompt with text inserted"""
        return cls.CURRENT_PROMPT.format(text=text)
    
    @classmethod
    def get_prompt_for_manual_use(cls) -> str:
        """Get prompt template for manual copy/paste workflow"""
        return cls.CURRENT_PROMPT.replace("{text}", "[PASTE YOUR TEXT HERE]")
    
    @classmethod
    def get_voice_assignment_info(cls) -> str:
        """Get voice assignment information for users"""
        return """Available Voices:
Male: Frank, Carter, Samuel, Anchen, Bowen
Female: Alice, Emily, Maya, Mary, Xinran

Voice Assignment:
- Speaker 0: Always uses Alice (narrator)
- Speaker 1, 2, 3...: Automatically assigned based on character gender
- Bob, Frank, etc. → Male voices (Frank, Carter)
- Alice, Emily, etc. → Female voices (Emily, Maya)"""
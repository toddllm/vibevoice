#!/usr/bin/env python3
"""
Simplified VibeVoice Demo - Works around Gradio API issues
"""

import gradio as gr
import torch
import numpy as np
import os
from pathlib import Path

# Import VibeVoice components
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

class SimpleVibeVoiceDemo:
    def __init__(self, model_path="microsoft/VibeVoice-1.5B"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        self.load_model()
        
    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"📥 Loading model from {self.model_path}")
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        # Load model with eager attention (no flash attention)
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def generate_speech(self, text, speaker_name="Alice"):
        """Generate speech from text."""
        try:
            # Simple generation for testing
            if not text.strip():
                return None, "❌ Please enter some text"
            
            # For now, return a simple message
            # In a full implementation, this would generate actual audio
            message = f"✅ Would generate speech for: '{text[:50]}...' with speaker {speaker_name}"
            
            # Create a dummy audio array for testing (1 second of silence)
            sample_rate = 24000
            duration = 1.0
            audio = np.zeros(int(sample_rate * duration))
            
            return (sample_rate, audio), message
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

def create_interface():
    """Create the Gradio interface."""
    demo = SimpleVibeVoiceDemo()
    
    with gr.Blocks(title="VibeVoice Simple Demo") as interface:
        gr.Markdown("# 🎙️ VibeVoice Simple Demo")
        gr.Markdown("Generate speech with VibeVoice (Simplified Interface)")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter Text",
                    placeholder="Type your text here...",
                    lines=3
                )
                
                speaker = gr.Dropdown(
                    choices=["Alice", "Bob", "Charlie"],
                    value="Alice",
                    label="Speaker"
                )
                
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
        
        # Connect the button
        generate_btn.click(
            fn=demo.generate_speech,
            inputs=[text_input, speaker],
            outputs=[audio_output, status_output]
        )
        
        # Add example
        gr.Examples(
            examples=[
                ["Hello, this is a test of VibeVoice.", "Alice"],
                ["Welcome to the demonstration.", "Bob"],
            ],
            inputs=[text_input, speaker]
        )
    
    return interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VibeVoice Simple Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    print("🚀 Starting VibeVoice Simple Demo...")
    interface = create_interface()
    
    # Disable API documentation to avoid the error
    interface.show_api = False
    
    interface.launch(
        server_port=args.port,
        server_name="0.0.0.0" if args.share else "127.0.0.1",
        share=args.share,
        show_api=False,  # Disable API docs
        quiet=False
    )
#!/usr/bin/env python3
"""
VibeVoice Flask POC - Minimal web interface for TTS generation
"""

import io
import os
import base64
import torch
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import tempfile
import traceback
import librosa

# Import VibeVoice
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

app = Flask(__name__)
CORS(app)

# Global model instance
model = None
processor = None
device = None

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VibeVoice POC</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        #status.success {
            background: #d4edda;
            color: #155724;
            display: block;
        }
        #status.error {
            background: #f8d7da;
            color: #721c24;
            display: block;
        }
        #status.loading {
            background: #fff3cd;
            color: #856404;
            display: block;
        }
        audio {
            width: 100%;
            margin-top: 20px;
        }
        .info {
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ VibeVoice POC</h1>
        
        <div>
            <label for="text">Enter text to synthesize:</label>
            <textarea id="text" rows="4" placeholder="Hello, this is a test of VibeVoice text-to-speech synthesis."></textarea>
        </div>
        
        <button id="generateBtn" onclick="generateSpeech()">Generate Speech</button>
        
        <div id="status"></div>
        
        <audio id="audioPlayer" controls style="display:none;"></audio>
        
        <div class="info">
            <strong>Status:</strong> <span id="modelStatus">Loading model...</span><br>
            <strong>Device:</strong> <span id="deviceInfo">Detecting...</span>
        </div>
    </div>
    
    <script>
        // Check model status on load
        window.onload = function() {
            checkStatus();
        };
        
        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('modelStatus').textContent = data.model_loaded ? 'Model loaded ✅' : 'Model loading...';
                    document.getElementById('deviceInfo').textContent = data.device || 'Unknown';
                    if (!data.model_loaded) {
                        setTimeout(checkStatus, 2000);
                    }
                })
                .catch(error => {
                    document.getElementById('modelStatus').textContent = 'Error checking status';
                    setTimeout(checkStatus, 5000);
                });
        }
        
        function generateSpeech() {
            const text = document.getElementById('text').value;
            const statusDiv = document.getElementById('status');
            const generateBtn = document.getElementById('generateBtn');
            const audioPlayer = document.getElementById('audioPlayer');
            
            if (!text.trim()) {
                statusDiv.className = 'error';
                statusDiv.textContent = 'Please enter some text';
                statusDiv.style.display = 'block';
                return;
            }
            
            // Show loading state
            statusDiv.className = 'loading';
            statusDiv.textContent = 'Generating speech... This may take a moment...';
            statusDiv.style.display = 'block';
            generateBtn.disabled = true;
            audioPlayer.style.display = 'none';
            
            // Send request
            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => {
                console.log('Response status:', response.status);
                console.log('Response headers:', response.headers);
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Generation failed');
                    });
                }
                return response.blob();
            })
            .then(blob => {
                console.log('Blob received:', blob.size, 'bytes, type:', blob.type);
                // Create audio URL and play
                const audioUrl = URL.createObjectURL(blob);
                console.log('Audio URL created:', audioUrl);
                audioPlayer.src = audioUrl;
                audioPlayer.style.display = 'block';
                
                statusDiv.className = 'success';
                statusDiv.textContent = 'Speech generated successfully! (' + blob.size + ' bytes)';
                
                // Auto-play with error handling
                audioPlayer.play().catch(e => {
                    console.error('Playback error:', e);
                    statusDiv.textContent += ' (Click play to start)';
                });
            })
            .catch(error => {
                statusDiv.className = 'error';
                statusDiv.textContent = 'Error: ' + error.message;
            })
            .finally(() => {
                generateBtn.disabled = false;
            });
        }
    </script>
</body>
</html>
"""

def load_model():
    """Load the VibeVoice model."""
    global model, processor, device
    
    print("🔄 Loading VibeVoice model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📍 Using device: {device}")
    
    try:
        # Load processor
        processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
        
        # Load model with eager attention
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            "microsoft/VibeVoice-1.5B",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            attn_implementation="eager",
        )
        model.eval()
        
        # Voice samples will be loaded per-request
        print("📝 Model ready for TTS generation")
        
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    """Check model status."""
    return jsonify({
        'model_loaded': model is not None,
        'device': device if device else 'Not initialized'
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate speech from text."""
    global model, processor
    
    if model is None:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"📝 Generating speech for: {text[:100]}...")
        
        # Format text with speaker label as expected by processor
        formatted_text = f"Speaker 0: {text}"
        
        # Load a default voice sample
        voice_path = "demo/voices/en-Alice_woman.wav"
        if os.path.exists(voice_path):
            print(f"📢 Loading voice: {voice_path}")
            wav, sr = sf.read(voice_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
            voice_sample = wav
        else:
            # Try alternative paths
            voice_paths = [
                "/home/tdeshane/VibeVoice/VibeVoice/demo/voices/en-Alice_woman.wav",
                "demo/voices/en-Frank_man.wav",
                "/home/tdeshane/VibeVoice/VibeVoice/demo/voices/en-Frank_man.wav"
            ]
            voice_sample = None
            for path in voice_paths:
                if os.path.exists(path):
                    print(f"📢 Loading voice: {path}")
                    wav, sr = sf.read(path)
                    if len(wav.shape) > 1:
                        wav = np.mean(wav, axis=1)
                    if sr != 24000:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
                    voice_sample = wav
                    break
            
            if voice_sample is None:
                return jsonify({'error': 'No voice samples found'}), 500
        
        # Process text with voice sample (both wrapped in lists for batch processing)
        inputs = processor(
            text=[formatted_text],  # Wrap in list
            voice_samples=[[voice_sample]],  # List of lists (batch of voice samples)
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate audio WITHOUT streaming (simpler approach)
        print("🎙️ Running model generation...")
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                tokenizer=processor.tokenizer,
                max_new_tokens=500,  # Reduced for stability
                do_sample=False,  # Disable sampling for deterministic generation
                cfg_scale=1.3,
                generation_config={
                    'do_sample': False,
                },
                # NO audio_streamer - get raw output
            )
        
        # Extract audio from the generation output
        print("🔊 Extracting generated audio...")
        if not hasattr(generation_output, 'speech_outputs') or not generation_output.speech_outputs:
            return jsonify({'error': 'Model did not generate speech_outputs'}), 500
        
        # Get the first speech output
        audio_tensor = generation_output.speech_outputs[0]
        audio = audio_tensor.cpu().numpy()
        print(f"✅ Got audio shape: {audio.shape}")
        
        # Ensure audio is in the right format
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Normalize audio
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.abs(audio).max()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, 24000)
            tmp_path = tmp_file.name
        
        # Send file
        response = send_file(
            tmp_path,
            mimetype='audio/wav',
            as_attachment=False
        )
        
        # Clean up after sending
        @response.call_on_close
        def cleanup():
            try:
                os.remove(tmp_path)
            except:
                pass
        
        return response
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting VibeVoice Flask POC...")
    
    # Load model in background
    load_model()
    
    # Start Flask app
    print("🌐 Starting web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
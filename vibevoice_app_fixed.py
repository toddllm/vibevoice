#!/usr/bin/env python3
"""
VibeVoice Flask app - FIXED with proper threading and AudioStreamer
"""

import io
import os
import threading
import time
import torch
import numpy as np
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import tempfile
import traceback

# Import VibeVoice components
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

app = Flask(__name__)
CORS(app)

# Global model instance
model = None
processor = None
device = None

# Simple HTML template (same as before)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VibeVoice TTS</title>
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
        <h1>🎙️ VibeVoice TTS (Fixed)</h1>
        
        <div>
            <label for="text">Enter text to synthesize:</label>
            <textarea id="text" rows="4" placeholder="Hello, this is a test of VibeVoice text-to-speech synthesis."></textarea>
        </div>
        
        <div style="margin-top: 15px;">
            <label for="duration">Max duration (seconds):</label>
            <input type="range" id="duration" min="10" max="300" value="60" oninput="updateDurationLabel()">
            <span id="durationLabel">60 seconds</span>
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
        window.onload = function() {
            checkStatus();
        };
        
        function updateDurationLabel() {
            const duration = document.getElementById('duration').value;
            document.getElementById('durationLabel').textContent = duration + ' seconds';
        }
        
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
            const duration = parseInt(document.getElementById('duration').value);
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
            statusDiv.textContent = 'Generating speech (up to ' + duration + ' seconds)... This may take a moment...';
            statusDiv.style.display = 'block';
            generateBtn.disabled = true;
            audioPlayer.style.display = 'none';
            
            // Send request
            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    text: text,
                    max_duration: duration
                })
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Generation failed');
                    });
                }
                return response.blob();
            })
            .then(blob => {
                const audioUrl = URL.createObjectURL(blob);
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
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"📍 Using device: {device}, dtype: {dtype}")
    
    try:
        # Load processor
        processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
        
        # Load model with eager attention
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            "microsoft/VibeVoice-1.5B",
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="eager",
        )
        model.eval()
        
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
    """Generate speech from text using proper threading and AudioStreamer."""
    global model, processor
    
    if model is None:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    try:
        data = request.json
        text = data.get('text', '').strip()
        max_duration = data.get('max_duration', 60)  # Default 60 seconds
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Calculate max tokens from duration (7.5 tokens per second)
        max_tokens = int(max_duration * 7.5)
        print(f"📝 Generating speech for: {text[:100]}...")
        print(f"⏱️ Max duration: {max_duration}s, Max tokens: {max_tokens}")
        
        # Format text with speaker label
        formatted_text = f"Speaker 0: {text}"
        
        # Load a default voice sample
        voice_paths = [
            "demo/voices/en-Alice_woman.wav",
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
        
        # Process inputs
        inputs = processor(
            text=[formatted_text],
            voice_samples=[[voice_sample]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Create AudioStreamer
        audio_streamer = AudioStreamer(
            batch_size=1,
            stop_signal=None,
            timeout=None
        )
        
        # Variable to store any exception from generation thread
        generation_exception = None
        
        def run_generation():
            """Run generation in background thread"""
            nonlocal generation_exception
            try:
                print("🎙️ Starting generation in background thread...")
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        tokenizer=processor.tokenizer,
                        audio_streamer=audio_streamer,  # Critical!
                        max_new_tokens=max_tokens,  # Dynamic based on user selection
                        do_sample=False,
                        cfg_scale=1.3,
                        generation_config={'do_sample': False}
                    )
                print("✅ Generation complete!")
            except Exception as e:
                generation_exception = e
                print(f"❌ Generation error: {e}")
            finally:
                audio_streamer.end()
        
        # Start generation in background thread
        generation_thread = threading.Thread(target=run_generation)
        generation_thread.start()
        
        # Give generation a moment to start
        time.sleep(0.5)
        
        # Collect audio chunks
        print("🔊 Collecting audio chunks...")
        audio_chunks = []
        chunk_count = 0
        
        # Get stream for first sample
        audio_stream = audio_streamer.get_stream(0)
        
        for audio_chunk in audio_stream:
            if audio_chunk is None:
                break
                
            # Convert to numpy if needed
            if torch.is_tensor(audio_chunk):
                if audio_chunk.dtype == torch.bfloat16:
                    audio_chunk = audio_chunk.float()
                audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            else:
                audio_np = audio_chunk
                
            chunk_count += 1
            print(f"  Chunk {chunk_count}: shape={audio_np.shape}")
            
            # Ensure 1D
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
                
            audio_chunks.append(audio_np)
        
        # Wait for generation to complete
        generation_thread.join()
        
        # Check for generation errors
        if generation_exception:
            raise generation_exception
        
        if not audio_chunks:
            return jsonify({'error': 'No audio generated'}), 500
        
        # Concatenate all chunks
        print(f"🎵 Concatenating {chunk_count} chunks...")
        audio = np.concatenate(audio_chunks)
        
        # Normalize if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / np.abs(audio).max()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, 24000)
            tmp_path = tmp_file.name
        
        print(f"✨ Generated {len(audio)/24000:.2f} seconds of audio")
        
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
    print("🚀 Starting VibeVoice Flask app (FIXED)...")
    
    # Load model
    load_model()
    
    # Start Flask app
    print("🌐 Starting web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
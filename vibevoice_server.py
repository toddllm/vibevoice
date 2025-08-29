#!/usr/bin/env python3
"""
VibeVoice LAN Server - Production deployment with REST API
"""

import os
import io
import time
import json
import base64
import threading
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import scipy.io.wavfile as wavfile
from vibevoice_production_v2 import VibeVoiceProductionServer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for LAN access

# Initialize VibeVoice server
vibevoice_server = VibeVoiceProductionServer()

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces for LAN access
PORT = 5000
DEFAULT_VOICE = 'demo/voices/en-Alice_woman.wav'

# =============================================================================
# Helper Functions
# =============================================================================

def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy array to WAV bytes"""
    buffer = io.BytesIO()
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    return buffer.read()

def create_sse_message(event: str, data: dict) -> str:
    """Create SSE message"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/')
def index():
    """Serve web interface or API documentation"""
    # Check if static/index.html exists
    static_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    if os.path.exists(static_path):
        return send_from_directory('static', 'index.html')
    else:
        # Fallback to API documentation
        return jsonify({
            'service': 'VibeVoice Production Server',
            'version': '2.0',
            'status': 'online',
            'endpoints': {
                '/health': 'GET - Health check and diagnostics',
                '/synthesize': 'POST - Generate speech (returns WAV)',
                '/synthesize/stream': 'POST - Stream speech generation (SSE)',
                '/models': 'GET - List available models',
                '/voices': 'GET - List available voices',
            },
            'models': ['1.5B', '7B'],
            'host': request.host,
        })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/advanced')
def advanced():
    """Serve advanced interface"""
    return send_from_directory('static', 'advanced.html')

@app.route('/pro')
def advanced_pro():
    """Serve professional advanced interface with all voices"""
    return send_from_directory('static', 'advanced_pro.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    health_data = vibevoice_server.health_check()
    health_data['timestamp'] = time.time()
    health_data['host'] = request.host
    return jsonify(health_data)

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'models': [
            {
                'id': '1.5B',
                'name': 'VibeVoice 1.5B',
                'transport': 'streaming',
                'description': 'Fast streaming model with lower latency'
            },
            {
                'id': '7B',
                'name': 'VibeVoice 7B',
                'transport': 'offline',
                'description': 'Higher quality offline generation'
            }
        ],
        'current': vibevoice_server.current_model_id
    })

@app.route('/voices', methods=['GET'])
def list_voices():
    """List available voice samples with detailed metadata"""
    from voice_manager import VoiceManager
    
    vm = VoiceManager('demo/voices')
    voices = []
    
    for voice_id, voice in vm.voices.items():
        voice_data = voice.to_dict()
        # Add additional metadata
        if voice_id in vm.BUILT_IN_VOICES:
            voice_data.update(vm.BUILT_IN_VOICES[voice_id])
        voices.append(voice_data)
    
    # Sort by language and then by name
    voices.sort(key=lambda x: (x['language'], x['speaker_name']))
    
    return jsonify({
        'voices': voices,
        'recommendations': vm.MULTI_SPEAKER_RECOMMENDATIONS,
        'total': len(voices)
    })

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """
    Main synthesis endpoint
    
    Request body:
    {
        "text": "Speaker 0: Hello world",
        "model": "1.5B" | "7B",
        "voice": "voice_id" (optional),
        "max_seconds": 30 (optional),
        "format": "wav" | "base64" (optional)
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({'error': 'text field is required'}), 400
        
        text = data['text']
        model_size = data.get('model', '1.5B')
        voice_id = data.get('voice')
        max_seconds = data.get('max_seconds', 30)
        output_format = data.get('format', 'wav')
        
        # Resolve voice path
        if voice_id:
            # Check if voice_id already includes .wav
            if voice_id.endswith('.wav'):
                voice_path = f"demo/voices/{voice_id}"
            else:
                voice_path = f"demo/voices/{voice_id}.wav"
            
            # Fallback to default if not found
            if not os.path.exists(voice_path):
                logger.warning(f"Voice not found: {voice_path}, using default")
                voice_path = DEFAULT_VOICE
        else:
            voice_path = DEFAULT_VOICE
        
        # Log request
        logger.info(f"Synthesis request: model={model_size}, len={len(text)}, voice={voice_path}")
        
        # Generate audio
        start_time = time.time()
        audio, metadata = vibevoice_server.generate(
            text=text,
            voice_path=voice_path,
            model_size=model_size,
            max_seconds=max_seconds
        )
        
        # Post-process
        audio = vibevoice_server.post_process_audio(audio)
        
        # Prepare response
        generation_time = time.time() - start_time
        
        if output_format == 'base64':
            # Return base64-encoded WAV
            wav_bytes = numpy_to_wav_bytes(audio)
            wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            return jsonify({
                'audio': wav_b64,
                'format': 'wav_base64',
                'sample_rate': 24000,
                'duration': metadata['duration_s'],
                'rtf': metadata['rtf'],
                'generation_time': generation_time,
                'model': model_size,
                'metadata': metadata
            })
        else:
            # Return WAV file directly
            wav_bytes = numpy_to_wav_bytes(audio)
            buffer = io.BytesIO(wav_bytes)
            buffer.seek(0)
            
            # Add metadata headers
            response = send_file(
                buffer,
                mimetype='audio/wav',
                as_attachment=False,
                download_name='synthesis.wav'
            )
            response.headers['X-Duration-Seconds'] = str(metadata['duration_s'])
            response.headers['X-RTF'] = str(metadata['rtf'])
            response.headers['X-Model'] = model_size
            response.headers['X-Generation-Time'] = str(generation_time)
            
            return response
            
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize/stream', methods=['POST'])
def synthesize_stream():
    """
    SSE streaming endpoint for real-time synthesis
    
    Only works with 1.5B model (streaming)
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'text field is required'}), 400
        
        text = data['text']
        voice_id = data.get('voice')
        
        # Force 1.5B for streaming
        model_size = '1.5B'
        
        # Resolve voice
        if voice_id:
            voice_path = f"demo/voices/{voice_id}.wav"
            if not os.path.exists(voice_path):
                voice_path = DEFAULT_VOICE
        else:
            voice_path = DEFAULT_VOICE
        
        def generate():
            """SSE generator"""
            try:
                # Send start event
                yield create_sse_message('start', {
                    'model': model_size,
                    'timestamp': time.time()
                })
                
                # Generate audio
                audio, metadata = vibevoice_server.generate(
                    text=text,
                    voice_path=voice_path,
                    model_size=model_size
                )
                
                # Post-process
                audio = vibevoice_server.post_process_audio(audio)
                
                # Convert to base64
                wav_bytes = numpy_to_wav_bytes(audio)
                wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                
                # Send audio event
                yield create_sse_message('audio', {
                    'data': wav_b64,
                    'duration': metadata['duration_s'],
                    'rtf': metadata['rtf'],
                    'chunks': metadata.get('chunks', 0)
                })
                
                # Send complete event
                yield create_sse_message('complete', {
                    'duration': metadata['duration_s'],
                    'timestamp': time.time()
                })
                
            except Exception as e:
                yield create_sse_message('error', {'message': str(e)})
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Server Management
# =============================================================================

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown endpoint (localhost only)"""
    if request.remote_addr not in ['127.0.0.1', '::1']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return jsonify({'status': 'shutting down'})

# =============================================================================
# Main
# =============================================================================

def main():
    """Start the server"""
    import socket
    
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print(" VibeVoice LAN Server Starting")
    print("="*60)
    print(f"  Local IP: {local_ip}")
    print(f"  Port: {PORT}")
    print(f"  Access URLs:")
    print(f"    - http://localhost:{PORT}")
    print(f"    - http://{local_ip}:{PORT}")
    print(f"    - http://{hostname}:{PORT}")
    print("="*60)
    print("  API Endpoints:")
    print("    GET  /              - API documentation")
    print("    GET  /health        - Health check")
    print("    POST /synthesize    - Generate speech")
    print("    POST /synthesize/stream - Stream speech (SSE)")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start Flask server
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,  # Enable threading for concurrent requests
        use_reloader=False
    )

if __name__ == '__main__':
    main()
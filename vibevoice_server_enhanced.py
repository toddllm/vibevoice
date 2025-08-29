#!/usr/bin/env python3
"""
Enhanced VibeVoice LAN Server with LLM and Voice Forge
"""

import os
import io
import time
import json
import base64
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
import logging

# Import enhanced server
from vibevoice_production_v3 import EnhancedVibeVoiceServer, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = load_config()

# Start with LLM disabled for safety
config['llm']['enabled'] = False
logger.info(f"LLM: {'ENABLED' if config['llm']['enabled'] else 'DISABLED'}")

# Initialize enhanced server
vibevoice_server = EnhancedVibeVoiceServer(config)

# Configuration
HOST = '0.0.0.0'
PORT = 5000
DEFAULT_VOICE = 'demo/voices/en-Alice_woman.wav'

# =============================================================================
# Helper Functions
# =============================================================================

def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy array to WAV bytes"""
    buffer = io.BytesIO()
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    return buffer.read()

def create_sse_message(event: str, data: dict) -> str:
    """Create SSE message"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# =============================================================================
# Web Interface Endpoints
# =============================================================================

@app.route('/')
def index():
    """Serve main interface or API documentation"""
    static_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    if os.path.exists(static_path):
        return send_from_directory('static', 'index.html')
    else:
        return jsonify({
            'service': 'Enhanced VibeVoice Server',
            'version': '3.0',
            'status': 'online',
            'features': {
                'llm': config['llm']['enabled'],
                'voice_forge': True,
                'models': ['1.5B', '7B']
            },
            'endpoints': {
                '/health': 'GET - Health check with enhanced metrics',
                '/synthesize': 'POST - Generate speech with optional LLM',
                '/process/text': 'POST - LLM text processing',
                '/voices': 'GET - List all voices including variants',
                '/voices/import': 'POST - Import custom voice',
                '/voices/variants': 'POST - Generate voice variants',
                '/studio': 'GET - Voice Studio interface'
            }
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
    """Serve professional interface"""
    return send_from_directory('static', 'advanced_pro.html')

@app.route('/studio')
def voice_studio():
    """Serve Voice Studio interface"""
    static_path = os.path.join('static', 'voice_studio.html')
    if os.path.exists(static_path):
        return send_from_directory('static', 'voice_studio.html')
    else:
        # Return a simple studio interface if file doesn't exist
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Voice Studio - Coming Soon</title>
            <style>
                body { font-family: Arial; padding: 20px; background: #1a1a2e; color: #eee; }
                .container { max-width: 800px; margin: auto; text-align: center; }
                h1 { color: #16a085; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎙️ Voice Studio</h1>
                <p>Voice import and variant creation interface coming soon!</p>
                <p>Use the API endpoints:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>POST /voices/import - Import custom voice</li>
                    <li>POST /voices/variants - Create variants</li>
                    <li>GET /voices - List all voices</li>
                </ul>
            </div>
        </body>
        </html>
        """

# =============================================================================
# Enhanced API Endpoints
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Enhanced health check with metrics"""
    health_data = vibevoice_server.health_check()
    health_data['timestamp'] = time.time()
    health_data['host'] = request.host
    health_data['enhanced_metrics'] = vibevoice_server.get_enhanced_metrics()
    return jsonify(health_data)

@app.route('/process/text', methods=['POST'])
def process_text():
    """LLM text processing endpoint"""
    data = request.get_json()
    text = data.get('text', '')
    force_llm = data.get('force_llm', False)
    
    if not vibevoice_server.llm_processor and force_llm:
        return jsonify({
            'error': 'LLM processor not enabled',
            'suggestion': 'Enable LLM in config or set force_llm=false'
        }), 503
    
    try:
        # Check messiness score
        from llm_text_processor import MessinessScorer
        scorer = MessinessScorer()
        messiness = scorer.score(text)
        
        # Process if messy enough or forced
        if force_llm or (vibevoice_server.llm_processor and messiness >= 0.35):
            if vibevoice_server.llm_processor:
                processed, units = vibevoice_server.llm_processor.process(text, force_llm)
                return jsonify({
                    'processed': processed,
                    'messiness_score': messiness,
                    'units': len(units),
                    'used_llm': True
                })
        
        # Use regex fallback
        from vibevoice_production_v2 import SpeakerParser
        parser = SpeakerParser()
        processed = parser.parse(text)
        
        return jsonify({
            'processed': processed,
            'messiness_score': messiness,
            'used_llm': False,
            'reason': 'Clean text, used regex parser'
        })
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Enhanced synthesis with optional LLM processing"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'text field is required'}), 400
        
        text = data['text']
        model_size = data.get('model', '1.5B')
        voice_id = data.get('voice')
        max_seconds = data.get('max_seconds', 30)
        output_format = data.get('format', 'wav')
        use_llm = data.get('use_llm', None)  # None = auto, True/False = forced
        
        # Resolve voice path
        if voice_id:
            if voice_id.endswith('.wav'):
                voice_path = f"demo/voices/{voice_id}"
            else:
                voice_path = f"demo/voices/{voice_id}.wav"
            
            # Check variants directory too
            if not os.path.exists(voice_path):
                variant_path = f"demo/voices/variants/{voice_id}.wav"
                if os.path.exists(variant_path):
                    voice_path = variant_path
                else:
                    logger.warning(f"Voice not found: {voice_path}, using default")
                    voice_path = DEFAULT_VOICE
        else:
            voice_path = DEFAULT_VOICE
        
        logger.info(f"Synthesis request: model={model_size}, voice={voice_path}, llm={use_llm}")
        
        # Generate audio
        start_time = time.time()
        audio, metadata = vibevoice_server.generate(
            text=text,
            voice_path=voice_path,
            model_size=model_size,
            max_seconds=max_seconds,
            use_llm=use_llm
        )
        
        generation_time = time.time() - start_time
        
        if output_format == 'base64':
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
                'used_llm': use_llm if use_llm is not None else 'auto',
                'metadata': metadata
            })
        else:
            wav_bytes = numpy_to_wav_bytes(audio)
            buffer = io.BytesIO(wav_bytes)
            buffer.seek(0)
            
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
            response.headers['X-Used-LLM'] = str(use_llm if use_llm is not None else 'auto')
            
            return response
            
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/voices', methods=['GET'])
def list_voices():
    """List all voices including imports and variants"""
    from voice_manager import VoiceManager
    import glob
    
    vm = VoiceManager('demo/voices')
    voices = []
    
    # Get built-in voices
    for voice_id, voice in vm.voices.items():
        voice_data = voice.to_dict()
        if voice_id in vm.BUILT_IN_VOICES:
            voice_data.update(vm.BUILT_IN_VOICES[voice_id])
        voice_data['type'] = 'built-in'
        voices.append(voice_data)
    
    # Add imported voices
    import_path = 'demo/voices/imports'
    if os.path.exists(import_path):
        for wav_file in glob.glob(f"{import_path}/*.wav"):
            voice_id = os.path.basename(wav_file).replace('.wav', '')
            voices.append({
                'id': voice_id,
                'path': wav_file,
                'name': voice_id,
                'type': 'imported',
                'display_name': f"📥 {voice_id}"
            })
    
    # Add variants
    variant_path = 'demo/voices/variants'
    if os.path.exists(variant_path):
        for wav_file in glob.glob(f"{variant_path}/*.wav"):
            voice_id = os.path.basename(wav_file).replace('.wav', '')
            voices.append({
                'id': voice_id,
                'path': wav_file,
                'name': voice_id,
                'type': 'variant',
                'display_name': f"🔄 {voice_id}"
            })
    
    return jsonify({
        'voices': voices,
        'total': len(voices),
        'types': {
            'built_in': sum(1 for v in voices if v.get('type') == 'built-in'),
            'imported': sum(1 for v in voices if v.get('type') == 'imported'),
            'variants': sum(1 for v in voices if v.get('type') == 'variant')
        },
        'recipes': vibevoice_server.voice_forge.list_available_recipes()
    })

@app.route('/voices/import', methods=['POST'])
def import_voice():
    """Import a custom voice"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Get metadata from form
    metadata = {
        'name': request.form.get('name', 'CustomVoice'),
        'lang': request.form.get('lang', 'en'),
        'gender': request.form.get('gender', 'neutral'),
        'consent': request.form.get('consent', 'false').lower() == 'true',
        'notes': request.form.get('notes', ''),
        'auto_variants': request.form.get('auto_variants', 'true').lower() == 'true'
    }
    
    if not metadata['consent']:
        return jsonify({'error': 'Consent required for voice import'}), 400
    
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name
        
        # Import voice
        result = vibevoice_server.import_voice(
            temp_path,
            metadata,
            auto_variants=metadata['auto_variants']
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Voice import error: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/voices/variants', methods=['POST'])
def create_variants():
    """Create voice variants"""
    data = request.get_json()
    
    base_voice = data.get('base_voice')
    recipes = data.get('recipes', ['bright_fast', 'warm_slow'])
    
    if not base_voice:
        return jsonify({'error': 'base_voice required'}), 400
    
    variants = []
    errors = []
    
    for recipe in recipes:
        try:
            variant_id = vibevoice_server.create_voice_variant(base_voice, recipe)
            variants.append({
                'recipe': recipe,
                'variant_id': variant_id,
                'status': 'created'
            })
        except Exception as e:
            errors.append({
                'recipe': recipe,
                'error': str(e)
            })
    
    return jsonify({
        'base_voice': base_voice,
        'variants': variants,
        'errors': errors,
        'success': len(variants) > 0
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'llm': {
            'enabled': config['llm']['enabled'],
            'engine': config['llm'].get('engine', 'none'),
            'auto_threshold': config['llm'].get('auto_threshold', 0.35)
        },
        'voice_forge': {
            'enabled': True,
            'recipes': list(vibevoice_server.voice_forge.variant_gen.RECIPES.keys())
        },
        'audio': {
            'crossfade_ms': config['audio'].get('crossfade_ms', 50),
            'max_pause_ms': config['audio'].get('max_pause_ms', 500)
        }
    })

@app.route('/config/llm', methods=['POST'])
def toggle_llm():
    """Enable/disable LLM processing"""
    data = request.get_json()
    enable = data.get('enable', False)
    
    config['llm']['enabled'] = enable
    
    # Reinitialize server if needed
    if enable and not vibevoice_server.llm_processor:
        from llm_text_processor import LLMTextProcessor
        vibevoice_server.llm_processor = LLMTextProcessor(config['llm'])
    
    return jsonify({
        'llm_enabled': config['llm']['enabled'],
        'message': f"LLM {'enabled' if enable else 'disabled'}"
    })

# =============================================================================
# Main
# =============================================================================

def main():
    """Start the enhanced server"""
    import socket
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*60)
    print(" Enhanced VibeVoice Server v3.0")
    print("="*60)
    print(f"  Local IP: {local_ip}")
    print(f"  Port: {PORT}")
    print(f"  Access URLs:")
    print(f"    - http://localhost:{PORT}")
    print(f"    - http://{local_ip}:{PORT}")
    print(f"    - http://{hostname}:{PORT}")
    print("="*60)
    print("  Web Interfaces:")
    print(f"    - http://{local_ip}:{PORT}/         - Main interface")
    print(f"    - http://{local_ip}:{PORT}/advanced - Advanced interface")
    print(f"    - http://{local_ip}:{PORT}/pro      - Professional interface")
    print(f"    - http://{local_ip}:{PORT}/studio   - Voice Studio")
    print("="*60)
    print("  Features:")
    print(f"    - LLM Processing: {'ENABLED' if config['llm']['enabled'] else 'DISABLED (safe mode)'}")
    print(f"    - Voice Forge: ENABLED")
    print(f"    - Models: 1.5B (streaming), 7B (offline)")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )

if __name__ == '__main__':
    main()
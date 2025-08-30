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

# Enable LLM for narrator functionality
config['llm']['enabled'] = True
config['llm']['model'] = 'qwen3:8b'  # Switch to 8B model for best text processing
logger.info(f"LLM: {'ENABLED' if config['llm']['enabled'] else 'DISABLED'}")

# Initialize enhanced server
vibevoice_server = EnhancedVibeVoiceServer(config)

# Force initialize LLM processor for qwen3 integration
if not vibevoice_server.llm_processor:
    from llm_text_processor import LLMTextProcessor
    vibevoice_server.llm_processor = LLMTextProcessor(config['llm'])
    logger.info("Force-initialized LLM processor for qwen3:8b")

# Initialize batch processor
from batch_processor import init_batch_processor
batch_processor = init_batch_processor(vibevoice_server.llm_processor, vibevoice_server)
logger.info("Batch processor initialized")

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

@app.route('/messy')
def messy_text_demo():
    """Messy text demo interface"""
    return send_from_directory('static', 'messy_text_demo.html')

@app.route('/normalize', methods=['POST'])
def normalize_text():
    """Normalize messy text using LLM if available, fallback to regex"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Use LLM processing only
        if vibevoice_server.llm_processor:
            processed_text, synthesis_units = vibevoice_server.llm_processor.process(text, force_llm=True)
            return jsonify({
                'normalized': processed_text,
                'units': len(synthesis_units),
                'used_llm': True
            })
        else:
            return jsonify({'error': 'LLM processor not available'}), 503
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/normalize/stream', methods=['POST'])
def normalize_stream():
    """Stream normalize results in real-time"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    def generate_stream():
        try:
            if vibevoice_server.llm_processor:
                text_length = len(text)
                word_count = len(text.split())
                yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {word_count} words with LLM (no timeout)...'})}\n\n"
                
                import time
                start_time = time.time()
                
                # Process with LLM - with progress updates for long requests
                try:
                    # Send periodic progress updates for long processing
                    def send_progress():
                        for i in range(30):  # Check for 30 seconds
                            time.sleep(1)
                            elapsed = time.time() - start_time
                            yield f"data: {json.dumps({'type': 'progress', 'message': f'Processing complex text... {elapsed:.0f}s elapsed'})}\n\n"
                    
                    import threading
                    import concurrent.futures
                    
                    # Start LLM processing in thread
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        llm_future = executor.submit(vibevoice_server.llm_processor.process, text, True)
                        
                        # Send progress updates while waiting
                        while not llm_future.done():
                            elapsed = time.time() - start_time
                            if elapsed > 10:  # After 10 seconds, send progress updates
                                yield f"data: {json.dumps({'type': 'progress', 'message': f'Still processing complex text... {elapsed:.0f}s elapsed'})}\n\n"
                            time.sleep(2)
                        
                        # Get result
                        processed_text, synthesis_units = llm_future.result(timeout=300)
                        duration = time.time() - start_time
                        
                        yield f"data: {json.dumps({'type': 'progress', 'message': f'LLM completed in {duration:.1f}s - Generated {len(synthesis_units)} units'})}\n\n"
                        
                        # Detailed logging for browser console
                        yield f"data: {json.dumps({'type': 'debug', 'message': f'Input length: {len(text)} chars, {len(text.split())} words'})}\n\n"
                        yield f"data: {json.dumps({'type': 'debug', 'message': f'Output length: {len(processed_text)} chars'})}\n\n"
                        yield f"data: {json.dumps({'type': 'debug', 'message': f'Speakers detected: {len(synthesis_units)} different speakers'})}\n\n"
                        
                        # Show first few lines for verification
                        preview_lines = processed_text.split('\\n')[:3]
                        yield f"data: {json.dumps({'type': 'debug', 'message': f'Preview: {preview_lines}'})}\n\n"
                        
                        # Log voice assignments if available
                        voice_assignments = getattr(vibevoice_server.llm_processor, '_last_voice_assignments', {})
                        if voice_assignments:
                            yield f"data: {json.dumps({'type': 'debug', 'message': f'Voice assignments: {voice_assignments}'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'debug', 'message': 'No explicit voice assignments - using defaults'})}\n\n"
                        
                        yield f"data: {json.dumps({'type': 'result', 'normalized': processed_text, 'units': len(synthesis_units), 'processing_time': duration})}\n\n"
                        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                        
                except concurrent.futures.TimeoutError:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Processing timed out - text too complex', 'error_type': 'timeout'})}\n\n"
                except Exception as llm_error:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'LLM processing failed: {str(llm_error)}', 'error_type': 'llm_processing'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'LLM processor not available', 'error_type': 'no_processor'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Server error: {str(e)}', 'error_type': 'server_error'})}\n\n"
    
    return Response(generate_stream(), mimetype='text/event-stream')

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

@app.route('/monitor')
def monitor():
    """Serve monitoring dashboard"""
    return send_from_directory('static', 'monitor.html')

# =============================================================================
# Enhanced API Endpoints
# =============================================================================

# =============================================================================
# Batch Processing Endpoints
# =============================================================================

@app.route('/batch/submit', methods=['POST'])
def submit_batch_job():
    """Submit text for batch processing"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text provided'}), 400
    
    job_id = batch_processor.submit_job(text)
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Job submitted for processing',
        'check_url': f'/batch/status/{job_id}'
    })

@app.route('/batch/status/<job_id>', methods=['GET'])
def get_batch_status(job_id):
    """Get status of a batch job"""
    job_status = batch_processor.get_job_status(job_id)
    
    if not job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_status)

@app.route('/batch/list', methods=['GET'])
def list_batch_jobs():
    """List all batch jobs"""
    jobs = batch_processor.list_jobs()
    return jsonify({'jobs': jobs})

@app.route('/batch/audio/<job_id>', methods=['GET'])
def get_batch_audio(job_id):
    """Get audio file for completed job"""
    job_status = batch_processor.get_job_status(job_id)
    
    if not job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_status['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    audio_file = job_status.get('audio_file')
    if not audio_file:
        return jsonify({'error': 'No audio file available'}), 404
    
    audio_path = os.path.join(batch_processor.results_dir, audio_file)
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    return send_file(audio_path, mimetype='audio/wav')

@app.route('/batch', methods=['GET'])
def batch_interface():
    """Serve simple batch processing interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>VibeVoice Batch Processing</title>
    <style>
        body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 20px; }
        .card { background: #f8f9fa; padding: 30px; border-radius: 10px; margin: 20px 0; }
        textarea { width: 100%; height: 200px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .status { margin: 20px 0; padding: 15px; border-radius: 5px; }
        .queued { background: #cce7ff; color: #004085; }
        .processing { background: #fff3cd; color: #856404; }
        .completed { background: #d4edda; color: #155724; }
        .failed { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>🎭 VibeVoice Batch Processing</h1>
    
    <div class="card">
        <h2>Submit Text for Processing</h2>
        <textarea id="inputText" placeholder="Paste your text here..."></textarea>
        <br><br>
        <button onclick="submitJob()">🚀 Submit Batch Job</button>
    </div>
    
    <div class="card">
        <h2>Job Status</h2>
        <div id="jobStatus">No jobs submitted yet</div>
        <button onclick="refreshStatus()" style="background: #28a745;">🔄 Refresh Status</button>
    </div>
    
    <script>
        let currentJobId = null;
        
        async function submitJob() {
            const text = document.getElementById('inputText').value.trim();
            if (!text) {
                alert('Please enter some text first');
                return;
            }
            
            try {
                const response = await fetch('/batch/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                
                const result = await response.json();
                currentJobId = result.job_id;
                
                document.getElementById('jobStatus').innerHTML = 
                    `<div class="status queued">Job ${result.job_id} submitted and queued for processing</div>`;
                
                // Start polling for status
                pollStatus();
                
            } catch (error) {
                document.getElementById('jobStatus').innerHTML = 
                    `<div class="status failed">Error: ${error.message}</div>`;
            }
        }
        
        async function refreshStatus() {
            if (!currentJobId) {
                document.getElementById('jobStatus').innerHTML = 'No job ID available';
                return;
            }
            
            try {
                const response = await fetch(`/batch/status/${currentJobId}`);
                const job = await response.json();
                
                displayJobStatus(job);
                
            } catch (error) {
                document.getElementById('jobStatus').innerHTML = 
                    `<div class="status failed">Error checking status: ${error.message}</div>`;
            }
        }
        
        function displayJobStatus(job) {
            const statusClass = job.status.replace('_', '');
            let html = `<div class="status ${statusClass}">
                <strong>Job ${job.job_id}</strong>: ${job.status}<br>
                <small>Submitted: ${new Date(job.created_at * 1000).toLocaleTimeString()}</small><br>
                <strong>Input Text:</strong> ${job.original_text_preview}
            </div>`;
            
            // Show processing times
            if (job.processing_time_text) {
                html += `<div><strong>✅ Text Processing:</strong> ${job.processing_time_text.toFixed(1)}s</div>`;
            }
            
            if (job.processing_time_audio) {
                html += `<div><strong>✅ Audio Processing:</strong> ${job.processing_time_audio.toFixed(1)}s</div>`;
            }
            
            // Show normalized text with better formatting
            if (job.normalized_text) {
                const lines = job.normalized_text.split('\\n');
                html += `<div style="margin: 20px 0;">
                    <strong>📝 Normalized Text (${lines.length} lines):</strong>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-line; max-height: 300px; overflow-y: auto;">${job.normalized_text}</div>
                </div>`;
            }
            
            // Show voice assignments with better formatting
            if (job.voice_assignments) {
                html += `<div style="margin: 20px 0;">
                    <strong>🎭 Voice Assignments:</strong>
                    <div style="background: #e8f4fd; padding: 15px; border-radius: 5px;">`;
                
                Object.entries(job.voice_assignments).forEach(([speaker, voice]) => {
                    const voiceName = voice.split('/').pop().replace('.wav', '').replace('en-', '').replace('_woman', '').replace('_man', '');
                    html += `<div>Speaker ${speaker}: ${voiceName}</div>`;
                });
                
                html += `</div></div>`;
            }
            
            // Show audio player
            if (job.audio_file) {
                html += `<div style="margin: 20px 0;">
                    <strong>🔊 Generated Audio:</strong><br>
                    <audio controls style="width: 100%; margin: 10px 0;" src="/batch/audio/${job.job_id}"></audio>
                    <div>Duration: ${job.audio_duration}s</div>
                </div>`;
            }
            
            // Show errors
            if (job.error_message) {
                html += `<div class="status failed" style="margin-top: 15px;">
                    <strong>❌ Error:</strong> ${job.error_message}
                </div>`;
            }
            
            document.getElementById('jobStatus').innerHTML = html;
        }
        
        function pollStatus() {
            if (!currentJobId) return;
            
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/batch/status/${currentJobId}`);
                    const job = await response.json();
                    
                    displayJobStatus(job);
                    
                    if (job.status === 'completed' || job.status === 'failed') {
                        clearInterval(interval);
                    }
                } catch (error) {
                    console.error('Polling error:', error);
                }
            }, 2000);
        }
    </script>
</body>
</html>
    """

@app.route('/prompt/current', methods=['GET'])
def get_current_prompt():
    """Get the current VibeVoice prompt template"""
    from prompt_templates import VibeVoicePrompts
    return jsonify({
        'prompt_template': VibeVoicePrompts.get_prompt_for_manual_use(),
        'voice_info': VibeVoicePrompts.get_voice_assignment_info(),
        'version': 'v1.0'
    })

@app.route('/debug/phi4', methods=['POST'])
def debug_phi4():
    """Debug endpoint to see raw phi4 output"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    try:
        if vibevoice_server.llm_processor:
            # Get raw phi4 output without parsing
            prompt = f"""Convert this text to VibeVoice format.

RULES:
- Text in quotes = dialogue (assign to specific speaker)
- Text not in quotes = narrative (assign to Speaker 0)
- Output EXACTLY one line per utterance in this format: "Speaker N: text"
- Use Speaker 0 for narrative/description
- Use Speaker 1, Speaker 2, etc. for named characters

EXAMPLE:
Input: "Hello," said Alice. She smiled.
Output:
Speaker 1: Hello
Speaker 0: said Alice. She smiled.

Now convert this text:
{text}

VibeVoice format output:"""
            
            # Direct engine call to see raw output
            raw_response = vibevoice_server.llm_processor.engine.generate(
                prompt,
                temperature=0.0,
                timeout=300  # 5 minute timeout for debugging
            )
            
            return jsonify({
                'input_text': text,
                'prompt_length': len(prompt),
                'raw_response': raw_response,
                'response_length': len(raw_response),
                'preview': raw_response[:500] + ('...' if len(raw_response) > 500 else '')
            })
        else:
            return jsonify({'error': 'LLM processor not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Enhanced health check with metrics"""
    health_data = vibevoice_server.health_check()
    health_data['timestamp'] = time.time()
    health_data['host'] = request.host
    
    # Add runtime metrics
    metrics = vibevoice_server.get_enhanced_metrics()
    metrics['total_requests'] = getattr(vibevoice_server, 'total_requests', 0)
    metrics['avg_rtf'] = getattr(vibevoice_server, 'avg_rtf', 2.5)
    
    # Add LLM cache stats if available
    if vibevoice_server.llm_processor:
        cache_stats = getattr(vibevoice_server.llm_processor, 'cache_stats', {})
        metrics['llm']['cache_hit_rate'] = cache_stats.get('hit_rate', 0)
    
    health_data['enhanced_metrics'] = metrics
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
        # Always use LLM processing - no fallbacks
        if vibevoice_server.llm_processor:
            processed, units = vibevoice_server.llm_processor.process(text, force_llm=True)
            return jsonify({
                'processed': processed,
                    'units': len(units),
                    'used_llm': True
                })
        else:
            return jsonify({'error': 'LLM processor not available'}), 503
        
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
        # If max_seconds is 0 or not provided, use model maximum
        if max_seconds == 0:
            # Model can handle up to ~15 minutes safely, but set reasonable default
            max_seconds = 900  # 15 minutes max
            logger.info(f"SERVER DEBUG: Converted max_seconds=0 to max_seconds={max_seconds}")
        else:
            logger.info(f"SERVER DEBUG: Using provided max_seconds={max_seconds}")
        output_format = data.get('format', 'wav')
        use_llm = data.get('use_llm', None)  # None = auto, True/False = forced
        
        # Resolve voice path
        if voice_id:
            try:
                from voice_utils import resolve_voice_path
                voice_path = str(resolve_voice_path(voice_id))
            except FileNotFoundError as e:
                logger.warning(f"Voice not found: {voice_id}, using default. {e}")
                voice_path = DEFAULT_VOICE
        else:
            voice_path = DEFAULT_VOICE
        
        logger.info(f"Synthesis request: model={model_size}, voice={voice_path}, llm={use_llm}")
        
        # Generate audio
        start_time = time.time()
        logger.info(f"SERVER DEBUG: About to call generate with max_seconds={max_seconds}")
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
    print(f"    - http://{local_ip}:{PORT}/monitor  - System Monitor")
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
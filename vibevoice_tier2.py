#!/usr/bin/env python3
"""
VibeVoice Tier 2 Implementation - Chunked generation with early-stop handling
"""

import io
import os
import time
import threading
import queue
import torch
import numpy as np
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, Response, render_template_string, send_file
from flask_cors import CORS
import tempfile
import traceback
import json
import re
from typing import List, Dict, Optional, Tuple, Generator

# Import VibeVoice components
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

app = Flask(__name__)
CORS(app)

# ========== CONSTANTS ==========
FRAME_RATE_HZ = 7.5
SAMPLE_RATE_HZ = 24000
SAMPLES_PER_FRAME = int(SAMPLE_RATE_HZ / FRAME_RATE_HZ)
FRAME_DURATION_MS = 1000 / FRAME_RATE_HZ
WORDS_PER_SECOND = 2.75

# Chunk defaults
DEFAULT_MAX_FRAMES_PER_CHUNK = 450  # ~60 seconds
DEFAULT_OVERLAP_FRAMES = 8  # ~1.07 seconds
MIN_CHUNK_FRAMES = 30  # ~4 seconds - avoid ultra-short utterances

# Global model instance
model = None
processor = None
device = None
memory_model = None
generation_limits = None

# ========== TIER 1 HELPERS (reused) ==========

def duration_to_frames(duration_seconds: float) -> int:
    frames = int(round(duration_seconds * FRAME_RATE_HZ))
    return max(frames, 1)

def frames_to_duration(frames: int) -> float:
    return frames / FRAME_RATE_HZ

def samples_to_frames(n_samples: int) -> int:
    return n_samples // SAMPLES_PER_FRAME

def est_frames_for_text(text: str) -> int:
    words = max(len(text.split()), 1)
    seconds = words / WORDS_PER_SECOND
    return duration_to_frames(seconds)

def split_sentences(text: str) -> List[str]:
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def count_text_tokens(processor, text: str) -> int:
    tokens = processor.tokenizer(text, add_special_tokens=False)
    return len(tokens['input_ids'])

# ========== TIER 2: TEXT-AWARE CHUNKING ==========

def extract_speaker(text: str) -> Optional[str]:
    """Extract speaker from text like 'Speaker 0: ...' or 'Alice: ...'"""
    match = re.match(r'^(Speaker \d+|[A-Za-z]+):\s*', text)
    return match.group(1) if match else None

def ensure_speaker(text: str, default_speaker: str = "Speaker 0") -> str:
    """Ensure text has a speaker label"""
    if not extract_speaker(text):
        return f"{default_speaker}: {text}"
    return text

class TextChunk:
    """Represents a chunk of text with metadata"""
    def __init__(self, text: str, estimated_frames: int, speaker: Optional[str] = None,
                 carryover_from_prev: bool = False):
        self.text = text
        self.estimated_frames = estimated_frames
        self.speaker = speaker or extract_speaker(text) or "Speaker 0"
        self.carryover_from_prev = carryover_from_prev
        self.actual_frames = 0  # Filled after generation

def chunk_text_with_speaker_safety(
    text: str,
    max_frames_per_chunk: int = DEFAULT_MAX_FRAMES_PER_CHUNK,
    min_frames_per_chunk: int = MIN_CHUNK_FRAMES
) -> List[TextChunk]:
    """
    Smart text chunking that respects speaker boundaries and frame budgets
    """
    # If no speaker labels found, add default speaker
    if not re.search(r'^(Speaker \d+|[A-Za-z]+):\s*', text, re.MULTILINE):
        text = f"Speaker 0: {text}"
    
    # Normalize names to Speaker N format for internal processing
    text = text.replace("Alice:", "Speaker 0:")
    text = text.replace("Bob:", "Speaker 1:")
    
    sentences = split_sentences(text)
    chunks = []
    current_chunk_sentences = []
    current_chunk_frames = 0
    current_speaker = "Speaker 0"  # Default speaker
    
    for sentence in sentences:
        # Check for speaker in this sentence
        sentence_speaker = extract_speaker(sentence)
        if sentence_speaker:
            # If speaker change detected and we have content, save current chunk
            if current_speaker and sentence_speaker != current_speaker and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(TextChunk(
                    ensure_speaker(chunk_text, current_speaker),
                    current_chunk_frames,
                    current_speaker
                ))
                # Start new chunk (NO carryover to avoid repetition)
                current_chunk_sentences = []
                current_chunk_frames = 0
            
            current_speaker = sentence_speaker
        
        # Estimate frames for this sentence
        sentence_frames = est_frames_for_text(sentence)
        
        # Check if adding this sentence would exceed limit
        if current_chunk_frames > 0 and current_chunk_frames + sentence_frames > max_frames_per_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(TextChunk(
                ensure_speaker(chunk_text, current_speaker),
                current_chunk_frames,
                current_speaker
            ))
            
            # Start new chunk (NO carryover to avoid repetition)
            current_chunk_sentences = [sentence]
            current_chunk_frames = sentence_frames
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_frames += sentence_frames
    
    # Add final chunk
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        # Inflate if too short
        if current_chunk_frames < min_frames_per_chunk and not chunks:
            # Single short chunk - add continuation hint to avoid BGM/singing
            chunk_text += " —"
        chunks.append(TextChunk(
            ensure_speaker(chunk_text, current_speaker),
            max(current_chunk_frames, min_frames_per_chunk),
            current_speaker
        ))
    
    return chunks

# ========== TIER 2: BUDGET CONTROLLER ==========

class BudgetController:
    """Controls generation budget and handles early stops"""
    def __init__(self, target_frames: int):
        self.target_frames = target_frames
        self.frames_generated = 0
        self.chunks_generated = []
        
    def should_continue(self) -> bool:
        """Check if we should generate more chunks"""
        return self.frames_generated < self.target_frames * 0.9  # 90% threshold
    
    def add_chunk(self, chunk_audio: np.ndarray, chunk: TextChunk):
        """Record a generated chunk"""
        actual_frames = samples_to_frames(len(chunk_audio))
        chunk.actual_frames = actual_frames
        # Note: frames_generated is now set directly from accumulator length
        self.chunks_generated.append((chunk_audio, chunk))
        
        # Log if chunk ended early
        if actual_frames < chunk.estimated_frames * 0.7:
            print(f"   ⚠️ Chunk ended early: {actual_frames}/{chunk.estimated_frames} frames")

# ========== TIER 2: CROSSFADE WITH SPEAKER SAFETY ==========

def crossfade_audio_safe(
    chunk1: np.ndarray,
    chunk2: np.ndarray,
    overlap_frames: int,
    speaker1: str,
    speaker2: str
) -> np.ndarray:
    """Crossfade with speaker boundary safety"""
    
    # No crossfade if speaker change
    if speaker1 != speaker2:
        print(f"   🎭 Speaker change: {speaker1} → {speaker2}, no crossfade")
        return np.concatenate([chunk1, chunk2], axis=0)
    
    # Standard crossfade for same speaker
    chunk1 = chunk1.astype(np.float32, copy=False)
    chunk2 = chunk2.astype(np.float32, copy=False)
    
    # Peak normalize to -1 dBFS
    for chunk in (chunk1, chunk2):
        peak = np.max(np.abs(chunk)) + 1e-9
        if peak > 0:
            chunk *= (0.8912509 / peak)
    
    overlap_samples = overlap_frames * SAMPLES_PER_FRAME
    
    if len(chunk1) < overlap_samples or len(chunk2) < overlap_samples:
        return np.concatenate([chunk1, chunk2], axis=0)
    
    # Equal-power crossfade
    fade_out = chunk1[-overlap_samples:]
    fade_in = chunk2[:overlap_samples]
    t = np.linspace(0, np.pi/2, overlap_samples, endpoint=True)
    blended = fade_out * np.cos(t) + fade_in * np.sin(t)
    
    return np.concatenate([
        chunk1[:-overlap_samples],
        blended,
        chunk2[overlap_samples:]
    ], axis=0)

# ========== TIER 2: CHUNKED SYNTHESIS ==========

def synthesize_single_chunk(
    chunk: TextChunk, voice_sample: np.ndarray,
    cfg_scale: float = 1.3, stop_event: Optional[threading.Event] = None
) -> Tuple[np.ndarray, int]:
    """Generate a single chunk, handling early stops gracefully"""
    
    global model, processor, device
    
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")
    
    # The processor ONLY accepts "Speaker N:" format where N is a number
    # We've already normalized to this format in chunking
    inputs = processor(
        text=[chunk.text],
        voice_samples=[[voice_sample]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    audio_streamer = AudioStreamer(
        batch_size=1,
        stop_signal=None,
        timeout=None
    )
    
    collected_audio = []
    frames_emitted = 0
    generation_exception = None
    
    def stop_check():
        return (frames_emitted >= chunk.estimated_frames) or \
               (stop_event and stop_event.is_set())
    
    def run_generation():
        nonlocal generation_exception
        try:
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    tokenizer=processor.tokenizer,
                    audio_streamer=audio_streamer,
                    cfg_scale=cfg_scale,
                    generation_config={'do_sample': False},
                    stop_check_fn=stop_check,
                    verbose=False,
                    refresh_negative=True,
                )
        except Exception as e:
            generation_exception = e
        finally:
            audio_streamer.end()
    
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()
    
    time.sleep(0.3)  # Let generation start
    
    audio_stream = audio_streamer.get_stream(0)
    for audio_chunk in audio_stream:
        if stop_event and stop_event.is_set():
            break
        if audio_chunk is None:
            break
            
        if torch.is_tensor(audio_chunk):
            if audio_chunk.dtype == torch.bfloat16:
                audio_chunk = audio_chunk.float()
            audio_np = audio_chunk.cpu().numpy().astype(np.float32)
        else:
            audio_np = audio_chunk.astype(np.float32)
        
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        
        collected_audio.append(audio_np)
        frames_emitted = samples_to_frames(sum(len(a) for a in collected_audio))
        
        # Early exit if we have enough
        if frames_emitted >= chunk.estimated_frames:
            break
    
    gen_thread.join(timeout=5.0)
    
    if generation_exception:
        raise generation_exception
    
    if not collected_audio:
        # Return short silence if nothing generated
        return np.zeros(MIN_CHUNK_FRAMES * SAMPLES_PER_FRAME, dtype=np.float32), MIN_CHUNK_FRAMES
    
    full_audio = np.concatenate(collected_audio, axis=0)
    
    # Normalize and clip
    peak = np.max(np.abs(full_audio)) + 1e-9
    if peak > 0:
        full_audio *= (0.8912509 / peak)
    full_audio = np.clip(full_audio, -1.0, 1.0).astype(np.float32)
    
    actual_frames = samples_to_frames(len(full_audio))
    return full_audio, actual_frames

def synthesize_chunked_with_budget(
    text: str, target_frames: int,
    max_frames_per_chunk: int = DEFAULT_MAX_FRAMES_PER_CHUNK,
    cfg_scale: float = 1.3, voice_samples: Optional[Dict[str, np.ndarray]] = None,
    progress_callback: Optional[callable] = None,
    stop_event: Optional[threading.Event] = None
) -> Generator[Dict, None, None]:
    """
    Chunked synthesis with budget control and SSE-ready yielding
    """
    # Chunk the text
    chunks = chunk_text_with_speaker_safety(text, max_frames_per_chunk)
    
    if not chunks:
        yield {'type': 'error', 'message': 'No chunks created from text'}
        return
    
    # Default voice samples
    if voice_samples is None:
        default_voice = load_default_voice()
        voice_samples = {"Speaker 0": default_voice}
    
    # Budget controller
    budget = BudgetController(target_frames)
    
    # Progress tracking
    start_time = time.time()
    
    yield {
        'type': 'started',
        'chunks_planned': len(chunks),
        'target_frames': target_frames,
        'target_seconds': frames_to_duration(target_frames)
    }
    
    # Use running accumulator pattern to avoid double-adding
    acc_audio = None
    prev_speaker = None
    
    for i, chunk in enumerate(chunks):
        if stop_event and stop_event.is_set():
            yield {'type': 'aborted'}
            break
        
        # Get voice for this speaker
        voice = voice_samples.get(chunk.speaker, list(voice_samples.values())[0])
        
        # Debug instrumentation
        print(f"\n📋 CHUNK[{i+1}/{len(chunks)}]")
        print(f"   Speaker: {chunk.speaker}")
        print(f"   Text: {chunk.text[:80]}..." if len(chunk.text) > 80 else f"   Text: {chunk.text}")
        print(f"   Est frames: {chunk.estimated_frames}")
        
        try:
            chunk_audio, actual_frames = synthesize_single_chunk(
                chunk, voice, cfg_scale, stop_event
            )
            
            print(f"   Generated: {len(chunk_audio)} samples, {actual_frames} frames, {len(chunk_audio)/SAMPLE_RATE_HZ:.2f}s")
            
            # Simple accumulator pattern
            same_speaker = (chunk.speaker == prev_speaker)
            
            if acc_audio is None:
                acc_audio = chunk_audio
                join_method = "init"
                overlap_frames = 0
            else:
                if same_speaker and DEFAULT_OVERLAP_FRAMES > 0:
                    # Crossfade for same speaker
                    acc_audio = crossfade_audio_safe(
                        acc_audio, chunk_audio, DEFAULT_OVERLAP_FRAMES,
                        prev_speaker, chunk.speaker
                    )
                    join_method = "crossfade"
                    overlap_frames = DEFAULT_OVERLAP_FRAMES
                else:
                    # Concatenate for different speakers
                    acc_audio = np.concatenate([acc_audio, chunk_audio], axis=0)
                    join_method = "concat"
                    overlap_frames = 0
            
            print(f"   JOIN: method={join_method}, overlap_frames={overlap_frames}")
            print(f"   ACC: {len(acc_audio)} samples, {samples_to_frames(len(acc_audio))} frames, {len(acc_audio)/SAMPLE_RATE_HZ:.2f}s")
            
            # Update budget based on actual accumulator length
            frames_so_far = samples_to_frames(len(acc_audio))
            budget.frames_generated = frames_so_far
            budget.chunks_generated.append((chunk_audio, chunk))
            
            # Progress update
            elapsed = time.time() - start_time
            rtf = frames_to_duration(frames_so_far) / elapsed if elapsed > 0 else 0
            
            yield {
                'type': 'chunk_done',
                'chunk_index': i,
                'chunk_frames': actual_frames,
                'total_frames': frames_so_far,
                'seconds_generated': frames_to_duration(frames_so_far),
                'rtf': rtf,
                'speaker': chunk.speaker
            }
            
            prev_speaker = chunk.speaker
            
            # Check if we've met our budget
            if not budget.should_continue():
                print(f"✅ Budget met: {frames_so_far}/{target_frames} frames")
                break
                
        except Exception as e:
            print(f"❌ Error in chunk {i}: {e}")
            import traceback
            traceback.print_exc()
            yield {'type': 'error', 'message': str(e), 'chunk_index': i}
            # Continue with next chunk instead of failing entirely
    
    # Final assembly
    if acc_audio is not None:
        full_audio = acc_audio
        
        elapsed = time.time() - start_time
        final_rtf = frames_to_duration(budget.frames_generated) / elapsed if elapsed > 0 else 0
        
        yield {
            'type': 'done',
            'audio': full_audio,
            'total_frames': budget.frames_generated,
            'total_seconds': frames_to_duration(budget.frames_generated),
            'rtf': final_rtf,
            'chunks_processed': len(budget.chunks_generated)
        }
    else:
        yield {'type': 'error', 'message': 'No audio generated'}

# ========== SSE STREAMING ==========

def generate_sse_stream(
    text: str, target_seconds: float,
    quality: str = "balanced", max_frames_per_chunk: int = DEFAULT_MAX_FRAMES_PER_CHUNK
) -> Generator[str, None, None]:
    """Generate Server-Sent Events stream"""
    
    target_frames = duration_to_frames(target_seconds)
    
    quality_presets = {
        'fast': {'cfg_scale': 1.0},
        'balanced': {'cfg_scale': 1.3},
        'high': {'cfg_scale': 1.5}
    }
    cfg_scale = quality_presets[quality]['cfg_scale']
    
    stop_event = threading.Event()
    
    for event in synthesize_chunked_with_budget(
        text, target_frames,
        max_frames_per_chunk, cfg_scale,
        stop_event=stop_event
    ):
        if event['type'] == 'done':
            # Save audio and send final event
            audio = event['audio']
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, SAMPLE_RATE_HZ)
                tmp_path = tmp.name
            
            yield f"data: {json.dumps({'type': 'done', 'wav_path': tmp_path, **{k: v for k, v in event.items() if k != 'audio'}})}\n\n"
        else:
            yield f"data: {json.dumps(event)}\n\n"

# ========== FLASK ROUTES ==========

@app.route('/synthesize_stream', methods=['GET'])
def synthesize_stream():
    """SSE endpoint for streaming synthesis"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Get parameters from query string for SSE
    text = request.args.get('script', '').strip()
    target_seconds = float(request.args.get('target_seconds', 60))
    quality = request.args.get('quality', 'balanced')
    
    if not text:
        return jsonify({'error': 'No script provided'}), 400
    
    return Response(
        generate_sse_stream(text, target_seconds, quality),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable Nginx buffering
        }
    )

@app.route('/get_temp_audio')
def get_temp_audio():
    """Serve temporary audio files"""
    import os
    path = request.args.get('path')
    if not path or not os.path.exists(path) or not path.startswith('/tmp/'):
        return jsonify({'error': 'Invalid path'}), 404
    
    return send_file(path, mimetype='audio/wav', as_attachment=False)

@app.route('/synthesize_chunked', methods=['POST'])
def synthesize_chunked_endpoint():
    """Regular endpoint that uses chunking internally"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.json
        text = data.get('script', '').strip()
        target_seconds = float(data.get('target_seconds', 60))
        quality = data.get('quality', 'balanced')
        
        if not text:
            return jsonify({'error': 'No script provided'}), 400
        
        target_frames = duration_to_frames(target_seconds)
        cfg_scale = {'fast': 1.0, 'balanced': 1.3, 'high': 1.5}[quality]
        
        # Collect all events
        events = []
        final_audio = None
        
        for event in synthesize_chunked_with_budget(
            text, target_frames,
            cfg_scale=cfg_scale
        ):
            # Don't include the raw audio array in events (not JSON serializable)
            if event['type'] == 'done' and 'audio' in event:
                final_audio = event['audio']
                # Create a copy without the audio array for JSON serialization
                event_copy = {k: v for k, v in event.items() if k != 'audio'}
                events.append(event_copy)
            else:
                events.append(event)
        
        if final_audio is None:
            return jsonify({'error': 'Generation failed', 'events': events}), 500
    except Exception as e:
        print(f"Error in synthesize_chunked: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    # Save and return
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, final_audio, SAMPLE_RATE_HZ)
        
        with open(tmp.name, 'rb') as f:
            wav_data = f.read()
        
        os.remove(tmp.name)
    
    import base64
    return jsonify({
        'wav_b64': base64.b64encode(wav_data).decode('utf-8'),
        'sample_rate': SAMPLE_RATE_HZ,
        'events': events
    })

# ========== MODEL LOADING (from Tier 1) ==========

def load_default_voice():
    """Load default voice sample"""
    voice_paths = [
        "demo/voices/en-Alice_woman.wav",
        "/home/tdeshane/VibeVoice/VibeVoice/demo/voices/en-Alice_woman.wav",
    ]
    
    for path in voice_paths:
        if os.path.exists(path):
            wav, sr = sf.read(path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
            return wav.astype(np.float32)
    
    return np.zeros(24000, dtype=np.float32)

def load_model():
    """Load model with memory calibration"""
    global model, processor, device, memory_model, generation_limits
    
    print("🔄 Loading VibeVoice model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"📍 Using device: {device}, dtype: {dtype}")
    
    try:
        processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
        
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
    """Test page with SSE support and audio playback"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>VibeVoice Tier 2 - Chunked + SSE</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; font-size: 14px; }
            button { margin: 10px 5px; padding: 10px 20px; font-size: 16px; }
            #output { background: #f0f0f0; padding: 10px; max-height: 300px; overflow-y: auto; }
            #audioContainer { margin-top: 20px; padding: 20px; background: #e0f0ff; border-radius: 8px; }
            audio { width: 100%; margin-top: 10px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
            .loading { background: #fff3cd; color: #856404; }
        </style>
    </head>
    <body>
        <h1>🎙️ VibeVoice Tier 2 - Multi-Speaker Synthesis</h1>
        <div>
            <label>Enter text (with optional speaker labels):</label>
            <textarea id="text" rows="10" placeholder="Enter text with multiple speakers...">
Alice: Hello everyone, welcome to our podcast.
Bob: Thanks for having me! I'm excited to discuss this topic.
Alice: Let's dive right in. What are your thoughts on the recent developments?
Bob: Well, I think there are several important points to consider here.
            </textarea>
        </div>
        <div>
            <label>Duration (seconds): </label>
            <input type="number" id="duration" value="60" min="5" max="300">
            <br>
            <button onclick="testSSE()">🔄 Stream with SSE</button>
            <button onclick="testChunked()">📦 Generate (Chunked)</button>
            <button onclick="clearOutput()">🗑️ Clear</button>
        </div>
        
        <div id="statusDiv"></div>
        
        <div id="audioContainer" style="display:none;">
            <h3>🎵 Generated Audio</h3>
            <audio id="audioPlayer" controls></audio>
            <div id="audioInfo"></div>
        </div>
        
        <h3>📊 Events Log</h3>
        <pre id="output"></pre>
        
        <script>
        function clearOutput() {
            document.getElementById('output').textContent = '';
            document.getElementById('statusDiv').innerHTML = '';
            document.getElementById('audioContainer').style.display = 'none';
        }
        
        function showStatus(message, type='loading') {
            const statusDiv = document.getElementById('statusDiv');
            statusDiv.innerHTML = '<div class="status ' + type + '">' + message + '</div>';
        }
        
        function testSSE() {
            const output = document.getElementById('output');
            const duration = document.getElementById('duration').value;
            output.textContent = 'Starting SSE stream...\\n';
            showStatus('Streaming audio generation...', 'loading');
            
            const params = new URLSearchParams({
                script: document.getElementById('text').value,
                target_seconds: duration,
                quality: 'balanced'
            });
            
            const evtSource = new EventSource('/synthesize_stream?' + params.toString());
            let audioPath = null;
            
            evtSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                output.textContent += JSON.stringify(data, null, 2) + '\\n';
                output.scrollTop = output.scrollHeight;
                
                if (data.type === 'chunk_done') {
                    showStatus('Processing chunk ' + (data.chunk_index + 1) + ', RTF: ' + data.rtf.toFixed(2) + 'x', 'loading');
                }
                
                if (data.type === 'done' && data.wav_path) {
                    evtSource.close();
                    audioPath = data.wav_path;
                    showStatus('Stream complete! Generated ' + data.total_seconds.toFixed(1) + 's, RTF: ' + data.rtf.toFixed(2) + 'x', 'success');
                    
                    // Load audio from temp file
                    fetch('/get_temp_audio?path=' + encodeURIComponent(audioPath))
                        .then(r => r.blob())
                        .then(blob => {
                            const audioUrl = URL.createObjectURL(blob);
                            const player = document.getElementById('audioPlayer');
                            player.src = audioUrl;
                            document.getElementById('audioContainer').style.display = 'block';
                            document.getElementById('audioInfo').textContent = 
                                'Generated: ' + data.total_seconds.toFixed(1) + 's | ' +
                                'Chunks: ' + data.chunks_processed + ' | ' +
                                'RTF: ' + data.rtf.toFixed(2) + 'x';
                        });
                }
                
                if (data.type === 'error') {
                    showStatus('Error: ' + data.message, 'error');
                }
            };
            
            evtSource.onerror = function(err) {
                evtSource.close();
                showStatus('Stream error occurred', 'error');
            };
        }
        
        function testChunked() {
            const output = document.getElementById('output');
            const duration = document.getElementById('duration').value;
            output.textContent = 'Generating audio...\\n';
            showStatus('Generating audio with chunking...', 'loading');
            
            fetch('/synthesize_chunked', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    script: document.getElementById('text').value,
                    target_seconds: parseInt(duration),
                    quality: 'balanced'
                })
            })
            .then(r => {
                if (!r.ok) throw new Error('Server error: ' + r.status);
                return r.json();
            })
            .then(data => {
                // Log events
                if (data.events) {
                    output.textContent += 'Events:\\n';
                    data.events.forEach(e => {
                        if (e.type !== 'done' || !e.audio) {  // Skip audio data in logs
                            output.textContent += JSON.stringify(e, null, 2) + '\\n';
                        }
                    });
                    output.scrollTop = output.scrollHeight;
                }
                
                // Play audio if available
                if (data.wav_b64) {
                    showStatus('Audio generated successfully!', 'success');
                    
                    // Convert base64 to blob
                    const byteCharacters = atob(data.wav_b64);
                    const byteNumbers = new Array(byteCharacters.length);
                    for (let i = 0; i < byteCharacters.length; i++) {
                        byteNumbers[i] = byteCharacters.charCodeAt(i);
                    }
                    const byteArray = new Uint8Array(byteNumbers);
                    const blob = new Blob([byteArray], {type: 'audio/wav'});
                    
                    // Create audio URL and play
                    const audioUrl = URL.createObjectURL(blob);
                    const player = document.getElementById('audioPlayer');
                    player.src = audioUrl;
                    document.getElementById('audioContainer').style.display = 'block';
                    
                    // Show info
                    const lastEvent = data.events.find(e => e.type === 'done');
                    if (lastEvent) {
                        document.getElementById('audioInfo').textContent = 
                            'Generated: ' + (lastEvent.total_seconds || 0).toFixed(1) + 's | ' +
                            'Chunks: ' + (lastEvent.chunks_processed || 0) + ' | ' +
                            'RTF: ' + (lastEvent.rtf || 0).toFixed(2) + 'x';
                    }
                } else {
                    showStatus('No audio generated', 'error');
                }
            })
            .catch(err => {
                output.textContent += '\\nError: ' + err;
                showStatus('Error: ' + err.message, 'error');
            });
        }
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Starting VibeVoice Tier 2...")
    load_model()
    print("🌐 Server ready on http://localhost:5003")
    print("   - POST /synthesize_chunked for regular chunked synthesis")
    print("   - GET /synthesize_stream for SSE streaming")
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
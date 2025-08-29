# VibeVoice API Documentation v3.0

## Base URL
```
http://localhost:5000
http://192.168.x.x:5000  # LAN access
```

## Authentication
Currently no authentication required (LAN deployment only)

## Core Endpoints

### 1. Speech Synthesis

#### `POST /synthesize`
Generate speech from text with optional LLM preprocessing.

**Request:**
```json
{
  "text": "Speaker 0: Hello world",
  "model": "1.5B",
  "voice": "en-Alice_woman",
  "max_seconds": 30,
  "use_llm": null,
  "format": "wav"
}
```

**Parameters:**
- `text` (required): Text to synthesize. Format: "Speaker N: text" or natural dialogue
- `model`: "1.5B" (streaming) or "7B" (offline). Default: "1.5B"
- `voice`: Voice ID. Default: "en-Alice_woman"
- `max_seconds`: Maximum output duration. Default: 30
- `use_llm`: true/false/null. null = auto-detect based on messiness
- `format`: "wav" or "base64". Default: "wav"

**Response (format: wav):**
- Binary WAV file
- Headers:
  - `X-Duration-Seconds`: Audio duration
  - `X-RTF`: Real-time factor
  - `X-Model`: Model used
  - `X-Generation-Time`: Processing time
  - `X-Used-LLM`: Whether LLM was used

**Response (format: base64):**
```json
{
  "audio": "base64_encoded_wav",
  "format": "wav_base64",
  "sample_rate": 24000,
  "duration": 2.5,
  "rtf": 0.25,
  "generation_time": 0.625,
  "model": "1.5B",
  "used_llm": false,
  "metadata": {}
}
```

### 2. Text Processing

#### `POST /process/text`
Process text with LLM to prepare for synthesis.

**Request:**
```json
{
  "text": "Alice: Hello!\nBob: Hi there!\n(pause)",
  "force_llm": false
}
```

**Parameters:**
- `text` (required): Raw text to process
- `force_llm`: Force LLM processing even for clean text

**Response:**
```json
{
  "processed": "Speaker 0: Hello!\nSpeaker 1: Hi there!",
  "messiness_score": 0.28,
  "units": 2,
  "used_llm": false,
  "reason": "Clean text, used regex parser"
}
```

### 3. Voice Management

#### `GET /voices`
List all available voices.

**Response:**
```json
{
  "voices": [
    {
      "id": "en-Alice_woman",
      "name": "en-Alice_woman",
      "display_name": "🇬🇧 Alice (Female)",
      "type": "built-in",
      "language": "en",
      "gender": "woman",
      "sample_rate": 16000,
      "duration": 9.27,
      "description": "Female English speaker, clear and professional",
      "best_for": "Podcasts, narration, general use",
      "characteristics": "Clear, friendly, versatile"
    }
  ],
  "total": 9,
  "types": {
    "built_in": 9,
    "imported": 0,
    "variants": 0
  },
  "recipes": {
    "bright_fast": "Bright & Fast",
    "warm_slow": "Warm & Slow",
    "neutral_mid": "Neutral Mid",
    "room_ambient": "Room Ambience"
  }
}
```

#### `POST /voices/import`
Import a custom voice.

**Request (multipart/form-data):**
- `audio`: Audio file (WAV/MP3/FLAC)
- `name`: Voice name
- `lang`: Language code (en/zh/es/fr/de/ja/ko/other)
- `gender`: neutral/woman/man
- `notes`: Optional description
- `consent`: "true" (required)
- `auto_variants`: "true"/"false"

**Response:**
```json
{
  "success": true,
  "voice_id": "en-CustomVoice_neutral",
  "voice_path": "demo/voices/imports/en-CustomVoice_neutral.wav",
  "variants": ["en-CustomVoice_neutral_bright_fast"],
  "segments": 5,
  "quality": {
    "duration": 45.2,
    "snr_db": 28.5,
    "rms_db": -22.3,
    "warnings": []
  }
}
```

#### `POST /voices/variants`
Create voice variants.

**Request:**
```json
{
  "base_voice": "en-Alice_woman",
  "recipes": ["bright_fast", "warm_slow"]
}
```

**Response:**
```json
{
  "base_voice": "en-Alice_woman",
  "variants": [
    {
      "recipe": "bright_fast",
      "variant_id": "en-Alice_woman_bright_fast",
      "status": "created"
    }
  ],
  "errors": [],
  "success": true
}
```

### 4. Configuration

#### `GET /config`
Get current configuration.

**Response:**
```json
{
  "llm": {
    "enabled": false,
    "engine": "ollama",
    "auto_threshold": 0.35
  },
  "voice_forge": {
    "enabled": true,
    "recipes": ["bright_fast", "warm_slow", "neutral_mid", "room_ambient"]
  },
  "audio": {
    "crossfade_ms": 50,
    "max_pause_ms": 500
  }
}
```

#### `POST /config/llm`
Enable/disable LLM processing.

**Request:**
```json
{
  "enable": true
}
```

**Response:**
```json
{
  "llm_enabled": true,
  "message": "LLM enabled"
}
```

### 5. Health & Monitoring

#### `GET /health`
Health check with metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1698765432.123,
  "host": "192.168.1.100:5000",
  "bf16_available": true,
  "current_model": "1.5B",
  "enhancements": {
    "llm_enabled": false,
    "voice_forge_ready": true,
    "available_recipes": ["bright_fast", "warm_slow"]
  },
  "enhanced_metrics": {
    "llm": {
      "llm_calls": 42,
      "fallbacks": 3,
      "cache_hits": 15,
      "avg_latency": 1.8
    },
    "voice_forge": {
      "voices_imported": 5,
      "variants_created": 12
    },
    "audio": {
      "silence_units_inserted": 28
    }
  }
}
```

## Error Responses

All endpoints may return error responses:

```json
{
  "error": "Error description"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `403`: Forbidden (e.g., consent not provided)
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable (e.g., LLM not enabled)

## Examples

### Basic Synthesis
```bash
curl -X POST http://localhost:5000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Speaker 0: Hello, this is a test.",
    "model": "1.5B"
  }' \
  -o output.wav
```

### Synthesis with LLM
```bash
curl -X POST http://localhost:5000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice: Welcome everyone!\nBob: Thanks for having me.\n(pause)",
    "model": "7B",
    "use_llm": true,
    "format": "base64"
  }'
```

### Import Voice
```bash
curl -X POST http://localhost:5000/voices/import \
  -F "audio=@myvoice.wav" \
  -F "name=MyVoice" \
  -F "lang=en" \
  -F "gender=neutral" \
  -F "consent=true" \
  -F "auto_variants=true"
```

### Create Variants
```bash
curl -X POST http://localhost:5000/voices/variants \
  -H "Content-Type: application/json" \
  -d '{
    "base_voice": "en-MyVoice_neutral",
    "recipes": ["bright_fast", "warm_slow"]
  }'
```

## Rate Limits

No rate limits in current LAN deployment. For production:
- Recommended: 10 requests/second per client
- Voice import: 1 per minute
- Variant creation: 5 per minute

## WebSocket Support (Future)

Planned for Stage 4:
- Real-time streaming synthesis
- Live voice modification
- Progressive audio delivery
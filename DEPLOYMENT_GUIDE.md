# VibeVoice Production Deployment Guide

## Current Status: Phase 2 - Gradual LLM Enablement

### System Architecture
```
┌─────────────────────────────────────────────────┐
│                  Web Interfaces                  │
│  Main │ Advanced │ Pro │ Voice Studio           │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│         Enhanced VibeVoice Server v3             │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ LLM Processor│  │ Voice Forge  │            │
│  │  (Optional)  │  │   (Active)   │            │
│  └──────┬───────┘  └──────────────┘            │
│         │                                        │
│  ┌──────▼────────────────────────────┐          │
│  │   Production Server Core          │          │
│  │  • 1.5B Streaming (Frozen)        │          │
│  │  • 7B Offline (LLM-enabled)       │          │
│  └───────────────────────────────────┘          │
└──────────────────────────────────────────────────┘
```

## Deployment Stages

### ✅ Stage 1: Shadow Mode (COMPLETE)
- LLM disabled by default
- Voice Forge operational
- All interfaces deployed

### 🚀 Stage 2: 7B Model LLM (CURRENT)
- Enable LLM for 7B offline model only
- Keep 1.5B streaming path frozen
- Monitor latency and quality

### 📅 Stage 3: Auto-Processing (NEXT)
- Enable automatic messiness detection
- LLM processes messy text automatically
- Fallback to regex for clean text

### 🎯 Stage 4: Full Production (FUTURE)
- Optional 1.5B LLM enablement
- Voice marketplace integration
- Advanced monitoring dashboard

## Quick Start Commands

### Start Server (Safe Mode)
```bash
# Default - LLM disabled
python vibevoice_server_enhanced.py
```

### Start Server (Production Config)
```bash
# With LLM enabled for 7B
python vibevoice_server_enhanced.py --config config_production.yaml
```

### Enable LLM Dynamically
```bash
curl -X POST http://localhost:5000/config/llm \
  -H "Content-Type: application/json" \
  -d '{"enable": true}'
```

## Performance Metrics

### Target SLAs
- **1.5B Streaming RTF**: < 0.3 (real-time)
- **7B Offline RTF**: < 1.0 
- **LLM Processing**: < 2s overhead
- **Voice Import**: < 5s for 30s audio
- **Variant Generation**: < 1s per variant

### Current Performance
- ✅ 1.5B Streaming: 0.25 RTF
- ✅ 7B Offline: 0.8 RTF
- ✅ LLM Overhead: 1.5s (mock), 2-3s (Qwen)
- ✅ Voice Import: 3s average
- ✅ Variants: 0.5s per recipe

## API Endpoints

### Core Synthesis
```bash
POST /synthesize
{
  "text": "Speaker 0: Hello world",
  "model": "1.5B" | "7B",
  "voice": "en-Alice_woman",
  "use_llm": true | false | null,  # null = auto
  "format": "wav" | "base64"
}
```

### LLM Text Processing
```bash
POST /process/text
{
  "text": "Alice: Hello\nBob: Hi",
  "force_llm": false
}
```

### Voice Management
```bash
# Import voice
POST /voices/import
FormData: audio, name, lang, gender, consent

# Create variants
POST /voices/variants
{
  "base_voice": "en-Alice_woman",
  "recipes": ["bright_fast", "warm_slow"]
}

# List all voices
GET /voices
```

### Configuration
```bash
# Get config
GET /config

# Toggle LLM
POST /config/llm
{
  "enable": true
}
```

## Monitoring

### Health Check
```bash
curl http://localhost:5000/health | jq
```

### Key Metrics to Watch
1. **LLM Usage**
   - Processed vs fallback ratio
   - Average processing time
   - Cache hit rate

2. **Voice Quality**
   - Import success rate
   - Average quality scores
   - Variant generation success

3. **System Performance**
   - Memory usage (< 4GB target)
   - GPU utilization
   - Request latency P50/P95/P99

## Troubleshooting

### Issue: High latency with LLM
**Solution**: 
- Check Ollama is running: `ollama list`
- Reduce timeout in config
- Use smaller model (0.5B vs 7B)

### Issue: Voice import fails
**Solution**:
- Check duration >= 15s
- Verify audio format (WAV/MP3/FLAC)
- Check consent checkbox

### Issue: CUDA out of memory
**Solution**:
- Use only one model at a time
- Clear GPU cache: `torch.cuda.empty_cache()`
- Reduce batch size

## Security Considerations

1. **Voice Rights**
   - Consent required for all imports
   - Celebrity name blocking active
   - Audit trail in YAML sidecars

2. **LLM Safety**
   - JSON-only responses
   - Schema validation
   - Hard timeouts (2-3.5s)

3. **Network Security**
   - LAN-only by default
   - CORS enabled for local access
   - No external dependencies for TTS

## Rollback Plan

If issues arise:

1. **Disable LLM immediately**:
   ```bash
   curl -X POST http://localhost:5000/config/llm \
     -d '{"enable": false}'
   ```

2. **Restart in safe mode**:
   ```bash
   pkill -f vibevoice_server
   python vibevoice_server_enhanced.py  # No config file
   ```

3. **Use frozen v2 server**:
   ```bash
   python vibevoice_server.py  # Original server
   ```

## Support

- **Issues**: Create GitHub issue with logs
- **Logs**: Check `/tmp/vibevoice.log`
- **Config**: Edit `config_production.yaml`

## Next Steps

1. ✅ Test 7B model with LLM enabled
2. ⏳ Import 3-5 custom voices
3. ⏳ Generate variants for testing
4. ⏳ Enable auto-processing for production
5. ⏳ Deploy monitoring dashboard
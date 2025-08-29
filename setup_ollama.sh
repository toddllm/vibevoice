#!/bin/bash
# Setup Ollama for VibeVoice LLM Processing

echo "=================================================="
echo " VibeVoice Ollama Setup"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service
echo "Starting Ollama service..."
ollama serve &> /dev/null &
sleep 2

# Pull Qwen models
echo "Pulling Qwen models for LLM processing..."

# Start with small model for testing
echo "1. Pulling Qwen 0.5B (test model)..."
ollama pull qwen2:0.5b

# Optional: Pull larger model for production
read -p "Pull Qwen 7B for production use? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "2. Pulling Qwen 7B (this may take a while)..."
    ollama pull qwen2:7b
fi

# Test the model
echo ""
echo "Testing Ollama integration..."
ollama run qwen2:0.5b "Convert to JSON: Alice says hello" --format json

echo ""
echo "=================================================="
echo " Ollama Setup Complete!"
echo "=================================================="
echo ""
echo "Available models:"
ollama list

echo ""
echo "To enable LLM in VibeVoice:"
echo "1. Edit config_production.yaml"
echo "2. Set llm.enabled: true"
echo "3. Restart the server with: python vibevoice_server_enhanced.py --config config_production.yaml"
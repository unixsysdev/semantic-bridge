# SemanticBridge 🌉

**Connect people through unexpected stories by finding the most distant-yet-connected ideas.**

Instead of traditional RAG (finding closest matches), SemanticBridge finds the *furthest apart* knowledge bites that still have a semantic thread connecting them. This creates surprising, serendipitous connections between people.

## Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r ../requirements.txt

# 2. Run the server
python main.py

# 3. Open the app
# Go to http://localhost:8000/app
```

## How It Works

1. **Choose Context**: Team brainstorm, strangers meeting, or couples/dates
2. **Select Mode**: 
   - 🌌 **Maximum Distance**: Find furthest-apart ideas with a thread
   - 🌉 **Surprise Bridge**: Pairwise surprising links (currently uses max distance)
   - 🧵 **Triplet Weave**: Three-way connection so everyone is included (3+ participants)
   - 🌐 **Centroid Constellation**: One idea per person around a shared center
   - ⛓️ **Bridge Chain**: A chain of ideas that includes all participants
   - 🎁 **Asymmetric Gift**: Something from one person illuminates another (couples only)
3. **Enter Knowledge Bites**: Memories, interests, random facts, obsessions
4. **Generate**: The app embeds all inputs, finds semantic distances, and generates stories

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /generate` - Main generation endpoint
- `POST /embed` - Debug endpoint for testing embeddings
- `GET /app` - Frontend UI

## Tech Stack

- **Backend**: FastAPI + Python
- **Embeddings**: Qwen3-Embedding-8B (via Chutes API)
- **LLM**: Qwen3-235B-A22B (via Chutes API)
- **Frontend**: Vanilla HTML/CSS/JS

## The Core Idea

Traditional RAG: "What's most similar?"  
SemanticBridge: "What's most distant but still connected?"

By finding knowledge bites that are far apart in embedding space but still have some semantic relationship, we surface unexpected connections that spark creativity, conversation, and connection between people.

## Mode Notes

- **Triplet Weave** requires at least 3 participants with bites.
- **Group modes for couples** (Centroid Constellation, Bridge Chain) require at least 2 bites per person.
- The app returns a JSON-formatted response from the LLM and shows optional debug insights about the embedding search in the UI.

## Settings

- **Temperature**: Higher = more creative/surprising output (0.1 - 1.5)
- **Stream**: Real-time response streaming (currently disabled)

---

Built for the Hangzhou hackathon 🚀

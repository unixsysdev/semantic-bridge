"""
SemanticBridge - Connect people through unexpected stories
FastAPI backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Get the directory paths
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

from embeddings import get_embeddings, get_single_embedding
from llm import generate_story, generate_stream
from semantic import (
    KnowledgeBite,
    find_maximum_distance_pairs,
    find_asymmetric_gifts,
    prepare_prompt_for_context
)

app = FastAPI(title="SemanticBridge", description="Connect people through unexpected stories")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Participant(BaseModel):
    id: str
    name: str
    bites: List[str]  # Their knowledge bites / memories / facts


class GenerateRequest(BaseModel):
    context: str  # "team", "strangers", "couples"
    mode: str  # "max_distance", "surprise_bridge", "asymmetric_gift"
    participants: List[Participant]
    temperature: Optional[float] = 0.8
    stream: Optional[bool] = False


class ConnectionResult(BaseModel):
    participant1: str
    bite1: str
    participant2: str
    bite2: str
    distance: float
    bridge_concept: Optional[str] = None


class GenerateResponse(BaseModel):
    connections: List[ConnectionResult]
    story: str


@app.get("/")
async def root():
    return {"message": "SemanticBridge API", "status": "ready", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/generate")
async def generate_connections(request: GenerateRequest):
    """
    Main endpoint: takes participants and their knowledge bites,
    finds semantic connections, generates story.
    """
    try:
        # Collect all bites and get embeddings
        all_bites: List[KnowledgeBite] = []
        all_texts: List[str] = []
        
        for participant in request.participants:
            for bite_text in participant.bites:
                all_bites.append(KnowledgeBite(
                    participant_id=participant.id,
                    participant_name=participant.name,
                    text=bite_text
                ))
                all_texts.append(bite_text)
        
        if len(all_texts) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 knowledge bites total")
        
        # Get embeddings for all bites
        embeddings = await get_embeddings(all_texts)
        
        # Attach embeddings to bites
        for bite, embedding in zip(all_bites, embeddings):
            bite.embedding = embedding
        
        # Find connections based on mode
        if request.mode == "max_distance":
            connections = find_maximum_distance_pairs(
                all_bites,
                cross_participant_only=True,
                top_k=3
            )
        elif request.mode == "asymmetric_gift" and len(request.participants) == 2:
            # For couples: find gifts from person 1 to person 2
            p1_bites = [b for b in all_bites if b.participant_id == request.participants[0].id]
            p2_bites = [b for b in all_bites if b.participant_id == request.participants[1].id]
            connections = find_asymmetric_gifts(p1_bites, p2_bites, top_k=3)
        else:
            # Default to max distance
            connections = find_maximum_distance_pairs(
                all_bites,
                cross_participant_only=True,
                top_k=3
            )
        
        if not connections:
            raise HTTPException(status_code=400, detail="Couldn't find interesting connections. Try adding more diverse knowledge bites!")
        
        # Prepare prompt and generate story
        prompt = prepare_prompt_for_context(connections, request.context, request.mode)
        
        if request.stream:
            # Return streaming response
            async def stream_generator():
                async for chunk in generate_stream(prompt, temperature=request.temperature):
                    yield chunk
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain"
            )
        else:
            story = await generate_story(prompt, temperature=request.temperature)
            
            # Format response
            connection_results = [
                ConnectionResult(
                    participant1=conn.bite1.participant_name,
                    bite1=conn.bite1.text,
                    participant2=conn.bite2.participant_name,
                    bite2=conn.bite2.text,
                    distance=conn.distance,
                    bridge_concept=conn.bridge_concept
                )
                for conn in connections
            ]
            
            return GenerateResponse(
                connections=connection_results,
                story=story
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def get_embedding(text: str):
    """Debug endpoint to test embeddings."""
    embedding = await get_single_embedding(text)
    return {"text": text, "embedding_dim": len(embedding), "sample": embedding[:5]}


# Serve frontend static files
from fastapi.responses import FileResponse

@app.get("/app")
@app.get("/app/{path:path}")
async def serve_frontend(path: str = ""):
    """Serve the frontend SPA."""
    if not path or path == "index.html":
        return FileResponse(FRONTEND_DIR / "index.html")
    
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    
    # Fallback to index.html for SPA routing
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

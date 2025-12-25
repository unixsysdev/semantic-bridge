"""
SemanticBridge - Connect people through unexpected stories
FastAPI backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv
import json
import logging
import re

load_dotenv()

# Logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("semanticbridge")

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
    find_triplet_connections,
    find_centroid_constellation,
    find_bridge_chain,
    prepare_prompt_for_context,
    prepare_prompt_for_group_context
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


def strip_think(text: str) -> tuple[str, Optional[str]]:
    """Remove <think>...</think> blocks and return (clean_text, think_text)."""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if matches:
        think_text = "\n\n".join(part.strip() for part in matches if part.strip())
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned, think_text or None
    if "<think>" in text:
        before, rest = text.split("<think>", 1)
        return before.strip(), rest.strip() or None
    return text, None


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


class GroupMember(BaseModel):
    participant: str
    bite: str


class GroupConnectionResult(BaseModel):
    members: List[GroupMember]
    score: float
    strategy: str


class GenerateResponse(BaseModel):
    connections: List[ConnectionResult]
    story: str
    reasoning: Optional[str] = None
    thinking: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    groups: Optional[List[GroupConnectionResult]] = None


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
        logger.info(
            "Generate request context=%s mode=%s participants=%d",
            request.context,
            request.mode,
            len(request.participants)
        )

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
        embedding_dim = len(embeddings[0]) if embeddings else 0
        logger.info("Embeddings fetched count=%d dim=%d", len(embeddings), embedding_dim)
        
        # Attach embeddings to bites
        for bite, embedding in zip(all_bites, embeddings):
            bite.embedding = embedding
        
        # Find connections based on mode
        search_debug: Dict[str, Any] = {}
        mode_used = request.mode
        mode_note = None
        connections = []
        groups = []
        pairing_strategy = "pairwise"

        if request.mode in {"triplet_weave", "centroid_constellation", "bridge_chain"}:
            pairing_strategy = "group"
            participant_ids = {p.id for p in request.participants}

            if request.mode == "triplet_weave" and len(participant_ids) < 3:
                raise HTTPException(status_code=400, detail="Triplet Weave requires at least 3 participants.")

            if request.context == "couples" and request.mode in {"centroid_constellation", "bridge_chain"}:
                for participant in request.participants:
                    bite_count = len([b for b in participant.bites if b.strip()])
                    if bite_count < 2:
                        raise HTTPException(
                            status_code=400,
                            detail="Group modes for couples require at least 2 knowledge bites per person."
                        )

            if request.mode == "triplet_weave":
                groups = find_triplet_connections(all_bites, top_k=3, debug=search_debug)
            elif request.mode == "centroid_constellation":
                groups = find_centroid_constellation(all_bites, top_k=1, debug=search_debug)
            elif request.mode == "bridge_chain":
                groups = find_bridge_chain(all_bites, top_k=1, debug=search_debug)

            if not groups:
                raise HTTPException(
                    status_code=400,
                    detail="Couldn't find a group connection. Try adding more diverse knowledge bites."
                )

            prompt = prepare_prompt_for_group_context(groups, request.context, request.mode)
        elif request.mode in {"max_distance", "surprise_bridge"}:
            connections = find_maximum_distance_pairs(
                all_bites,
                cross_participant_only=True,
                top_k=3,
                debug=search_debug
            )
            if not connections:
                raise HTTPException(
                    status_code=400,
                    detail="Couldn't find interesting connections. Try adding more diverse knowledge bites!"
                )
            prompt = prepare_prompt_for_context(connections, request.context, request.mode)
        elif request.mode == "asymmetric_gift" and len(request.participants) == 2:
            # For couples: find gifts from person 1 to person 2
            p1_bites = [b for b in all_bites if b.participant_id == request.participants[0].id]
            p2_bites = [b for b in all_bites if b.participant_id == request.participants[1].id]
            connections = find_asymmetric_gifts(p1_bites, p2_bites, top_k=3, debug=search_debug)
            if not connections:
                raise HTTPException(
                    status_code=400,
                    detail="Couldn't find interesting connections. Try adding more diverse knowledge bites!"
                )
            prompt = prepare_prompt_for_context(connections, request.context, request.mode)
        else:
            # Default to max distance
            connections = find_maximum_distance_pairs(
                all_bites,
                cross_participant_only=True,
                top_k=3,
                debug=search_debug
            )
            if not connections:
                raise HTTPException(
                    status_code=400,
                    detail="Couldn't find interesting connections. Try adding more diverse knowledge bites!"
                )
            prompt = prepare_prompt_for_context(connections, request.context, "max_distance")
            if request.mode != "max_distance":
                mode_used = "max_distance"
                mode_note = f"Mode '{request.mode}' is not available for this request; fell back to max_distance."
                logger.info(mode_note)
        
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
            raw_story = await generate_story(prompt, temperature=request.temperature)
            cleaned_story, thinking = strip_think(raw_story)
            story = cleaned_story
            reasoning = None
            # Try to parse JSON response with ideas + reasoning
            cleaned = cleaned_story.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            try:
                parsed = json.loads(cleaned)
                story = parsed.get("ideas", story)
                reasoning = parsed.get("reasoning")
            except json.JSONDecodeError:
                try:
                    start = cleaned.find("{")
                    end = cleaned.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(cleaned[start:end + 1])
                        story = parsed.get("ideas", story)
                        reasoning = parsed.get("reasoning")
                    else:
                        logger.warning("LLM response did not contain JSON; returning raw text")
                except json.JSONDecodeError:
                    logger.warning("LLM response was not valid JSON; returning raw text")
            
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

            group_results = [
                GroupConnectionResult(
                    members=[
                        GroupMember(participant=member.participant_name, bite=member.text)
                        for member in group.members
                    ],
                    score=group.score,
                    strategy=group.strategy
                )
                for group in groups
            ] if groups else None
            
            debug_info = {
                "context": request.context,
                "mode_requested": request.mode,
                "mode_used": mode_used,
                "mode_note": mode_note,
                "participant_count": len(request.participants),
                "bite_count": len(all_bites),
                "embedding": {
                    "model": "Qwen3-Embedding-8B",
                    "dimensions": embedding_dim
                },
                "pairing_strategy": pairing_strategy,
                "group_count": len(group_results) if group_results else 0,
                "search": search_debug
            }
            logger.info("Search debug: %s", debug_info)

            return GenerateResponse(
                connections=connection_results,
                story=story,
                reasoning=reasoning,
                thinking=thinking,
                debug=debug_info,
                groups=group_results
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

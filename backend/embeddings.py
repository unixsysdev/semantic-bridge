"""
Chutes API client for embeddings using Qwen3-Embedding-8B
"""
import httpx
import os
from typing import List

CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")
EMBEDDING_URL = "https://chutes-qwen-qwen3-embedding-8b.chutes.ai/v1/embeddings"


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using Qwen3-Embedding-8B.
    Returns a list of embedding vectors.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        embeddings = []
        for text in texts:
            response = await client.post(
                EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {CHUTES_API_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": None
                }
            )
            response.raise_for_status()
            data = response.json()
            # Extract embedding from response
            embedding = data["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings


async def get_single_embedding(text: str) -> List[float]:
    """Get embedding for a single text."""
    embeddings = await get_embeddings([text])
    return embeddings[0]

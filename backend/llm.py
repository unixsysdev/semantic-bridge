"""
Chutes API client for LLM generation using Qwen3-235B
"""
import httpx
import os
from typing import Optional

CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")
LLM_URL = "https://llm.chutes.ai/v1/chat/completions"


async def generate_story(
    prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 1024
) -> str:
    """
    Generate text using Qwen3-235B.
    Higher temperature = more creative/surprising.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {CHUTES_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Qwen/Qwen3-235B-A22B",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a creative storyteller and connection-finder. You find unexpected, meaningful bridges between disparate ideas. You write with warmth, surprise, and a sense of magic. Keep responses concise but evocative. Follow the response format requested by the user and output JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def generate_stream(
    prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 1024
):
    """
    Stream generation for real-time display.
    Yields chunks of text as they arrive.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            LLM_URL,
            headers={
                "Authorization": f"Bearer {CHUTES_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Qwen/Qwen3-235B-A22B",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a creative storyteller and connection-finder. You find unexpected, meaningful bridges between disparate ideas. You write with warmth, surprise, and a sense of magic. Keep responses concise but evocative. Follow the response format requested by the user and output JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except:
                        pass

import os
import asyncio
import json
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('127.0.0.1')

url = 'https://openrouter.ai/api/v1/chat/completions'

api_key = os.getenv('OPENROUTER_API_KEY')

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

base_prompt = f"""
You are an AI agent who can understand the injected data and train yourself using that data.
You do NOT update your weights while training.
You MUST train on the data provided to you.
"""
@mcp.tool()
async def llm_call(prompt:str):
    prompt = base_prompt
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                f"role": "user",
                f"content": {prompt}
            }
        ]
    }
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.post(url, headers=headers, json=data)

            if response.status_code== 429:
                return{
                    "error":"RATE_LIMITED",
                    "status":429
                }
            if response.status_code != 200:
                return {
                    "error":"LLM_HTTP_ERROR",
                    "status":response.status_code,
                    "body":response.text
                }
        except httpx.TimeoutException:
                return {
                    "error": "TIMEOUT"
                }

        except Exception as e:
            return {
               "error": "LLM_CALL_FAILED",
                "details": str(e)
            }
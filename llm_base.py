import os
import asyncio
import httpx
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('AgentSociety')

url = 'https://openrouter.ai/api/v1/chat/completions'

api_key = os.getenv('OPENROUTER_API_KEY')

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

prompt = f"""
You are a helpful assistant.
"""
async def mistralai(prompt):
    data = {
        'model': 'mistralai/devstral-2512:free',
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response

async def nemotron(prompt):
    data = {
        'model': 'nvidia/nemotron-nano-12b-v2-vl:free',
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response
    
async def openai(prompt):
    data = {
        'model': 'openai/gpt-oss-120b:free',
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response
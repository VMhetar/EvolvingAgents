import os
import json
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("127.0.0.1")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

BASE_PROMPT = """
You are a frozen reasoning engine.

You DO NOT update model weights.
You DO NOT store raw data.
You ONLY propose structured state updates.

Given current_state and new_data,
return ONLY valid JSON in this format:

{
  "state_delta": {
    "key": "value"
  },
  "confidence": 0.0
}

No explanations.
No extra text.
"""

@mcp.tool()
async def llm_call(prompt: str):
    final_prompt = BASE_PROMPT + "\n\n" + prompt

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        "temperature": 0.0
    }

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.post(
                OPENROUTER_URL,
                headers=HEADERS,
                json=payload
            )

            if response.status_code == 429:
                return {"error": "RATE_LIMITED"}

            if response.status_code != 200:
                return {
                    "error": "LLM_HTTP_ERROR",
                    "status": response.status_code,
                    "body": response.text
                }

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return {
                    "error": "INVALID_JSON_FROM_LLM",
                    "raw": content
                }

            return parsed

        except httpx.TimeoutException:
            return {"error": "TIMEOUT"}

        except Exception as e:
            return {
                "error": "LLM_CALL_FAILED",
                "details": str(e)
            }

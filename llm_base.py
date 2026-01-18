import os
import json
import httpx
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("127.0.0.1")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MODEL_ID = "meta-llama/llama-3.1-8b-instruct"

SYSTEM_PROMPT = """
You are a frozen reasoning engine.

Rules (non-negotiable):
- You DO NOT update model weights.
- You DO NOT store raw observations.
- You ONLY output structured state updates.
- You MUST return valid JSON.
- No explanations. No prose. No markdown.

Output schema:

{
  "state_delta": { "key": "value" },
  "confidence": <float between 0 and 1>
}

If no update is justified, return:

{
  "state_delta": {},
  "confidence": 0.0
}
""".strip()


def _validate_response(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Response is not a JSON object")

    if "state_delta" not in obj or "confidence" not in obj:
        raise ValueError("Missing required fields")

    if not isinstance(obj["state_delta"], dict):
        raise ValueError("state_delta must be a dict")

    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        raise ValueError("confidence must be a float between 0 and 1")

    return {
        "state_delta": obj["state_delta"],
        "confidence": float(conf)
    }


@mcp.tool()
async def llm_call(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
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

            raw = response.json()["choices"][0]["message"]["content"]

            try:
                parsed = json.loads(raw)
                return _validate_response(parsed)

            except Exception as e:
                return {
                    "error": "INVALID_MODEL_OUTPUT",
                    "raw": raw,
                    "details": str(e)
                }

        except httpx.TimeoutException:
            return {"error": "TIMEOUT"}

        except Exception as e:
            return {
                "error": "LLM_CALL_FAILED",
                "details": str(e)
            }

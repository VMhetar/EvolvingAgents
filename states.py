import json
from typing import Dict, Any
from llm_base import llm_call

ALLOWED_KEYS = {"concepts", "rules", "summaries", "constraints"}

class State:
    def __init__(self, state: Dict[str, Any]):
        self.state = state

    def get_state(self) -> Dict[str, Any]:
        return self.state

    async def update_state(self, new_data: Dict[str, Any]):
        prompt = f"""
Current state:
{json.dumps(self.state, indent=2)}

New data:
{json.dumps(new_data, indent=2)}

Return minimal state_delta.
"""

        update_data = await llm_call(prompt)

        if "state_delta" not in update_data:
            return self.state

        if update_data.get("confidence", 0) < 0.6:
            return self.state

        for key, value in update_data["state_delta"].items():
            if key in ALLOWED_KEYS:
                self.state[key] = value

        self.state["_version"] = self.state.get("_version", 0) + 1
        return self.state

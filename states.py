import json
from typing import Dict, Any, List
from llm_base import llm_call

STATE_SCHEMA = {
    "concepts": list, 
    "rules": list, 
    "summaries": list,  
    "constraints": list 
}

CONFIDENCE_THRESHOLD = 0.6
MAX_ITEMS_PER_KEY = 50 


class State:
    def __init__(self, state: Dict[str, Any] | None = None):
        self.state = state or {k: [] for k in STATE_SCHEMA}
        self.state.setdefault("_version", 0)

    def get_state(self) -> Dict[str, Any]:
        return self.state

    async def update_state(self, new_data: Dict[str, Any]):
        prompt = f"""
Current state (compressed belief store):
{json.dumps(self.state, indent=2)}

New observation:
{json.dumps(new_data, indent=2)}

Propose minimal state_delta.
Only include changed or new beliefs.
"""

        update_data = await llm_call(prompt)

        if (
            not isinstance(update_data, dict)
            or update_data.get("confidence", 0) < CONFIDENCE_THRESHOLD
            or "state_delta" not in update_data
        ):
            return self.state

        delta = update_data["state_delta"]

        for key, value in delta.items():
            if key not in STATE_SCHEMA:
                continue

            expected_type = STATE_SCHEMA[key]

            # Normalize to list
            if isinstance(value, expected_type):
                new_items = value
            else:
                new_items = [value]

            if not isinstance(new_items, list):
                continue

            # Append, do not overwrite
            for item in new_items:
                if item not in self.state[key]:
                    self.state[key].append(item)

            # Memory pressure: trim oldest
            if len(self.state[key]) > MAX_ITEMS_PER_KEY:
                self.state[key] = self.state[key][-MAX_ITEMS_PER_KEY:]

        self.state["_version"] += 1
        return self.state

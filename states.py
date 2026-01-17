from llm_base import llm_call
from typing import Dict, List, Any

class State:
    def __init__(self, state: Dict[str, Any]):
        self.state = state

    def get_state(self)-> Dict[str, Any]:
        return self.state
    
    async def update_state(self, state: Dict[str, Any],prompt:str):
        update_data = await llm_call(prompt)
        self.state.update(update_data)
        return self.state
    
import json
from typing import Dict, List, Any
from llm_base import llm_call

class PersistentMemory:
    def __init__(self, memory: List[Dict[str, Any]] | None = None):
        self.memory = memory or []
    
    def get_memory(self):
        return self.memory
    
    async def propose_memory_update(self, new_data: Dict[str, Any]):
        prompt = f"""
        Current memory (compressed experiences):
        {json.dumps(self.memory[-10:], indent=2)}

        New observation:
        {json.dumps(new_data, indent=2)}

        If this observation is worth remembering, propose a SHORT memory entry.
        If not, return null.

        Output JSON only:
        {{
        "memory_entry": {{ ... }} | null,
        "confidence": <0 to 1>
        }}
        """

        result = await llm_call(prompt)
        return result
    
    def commit(self, proposal:Dict[str, Any]):
        if (not proposal or proposal.get("confidence", 0) < 0.6 or not proposal.get("memory_entry")):
            return 
        self.memory.append(proposal["memory_entry"])

        if len(self.memory) > 200:
            self.memory = self.memory[-200:]
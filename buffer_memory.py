from typing import Dict, List, Any
import time

class BufferMemory:
    def __init__(self, max_size:int=20, ttl_seconds:int=300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.buffer: List[Dict[str,Any]] = []

    def add(self, observation:Dict[str, Any]):
        entry = {
            "timestamps": time.time(),
            "data":observation
        }
        self.buffer.append(entry)
        self._prune()

    def get(self) -> List[Dict[str, Any]]:
        self._prune()
        return [e["data"] for e in self.buffer]

    def clear(self):
        self.buffer = []

    def _prune(self):
        now = time.time()

        self.buffer = [e for e in self.buffer if now - e["timestamp"] <= self.ttl]
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
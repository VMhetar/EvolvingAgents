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
        cleaned = []

        for e in self.buffer:
            # Drop malformed entries safely
            if not isinstance(e, dict):
                continue
            if "timestamp" not in e or "data" not in e:
                continue

            if now - e["timestamp"] <= self.ttl:
                cleaned.append(e)

    # Keep most recent entries only
        if len(cleaned) > self.max_size:
            cleaned = cleaned[-self.max_size:]

        self.buffer = cleaned

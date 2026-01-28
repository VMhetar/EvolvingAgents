import json
from buffer_memory import BufferMemory
from persistent_memory import PersistentMemory
from states import State
from llm_base import llm_call


# --- Initialize cognitive layers ---
buffer = BufferMemory(max_size=20, ttl_seconds=300)
memory = PersistentMemory()
state = State()


async def cognitive_step(observation: dict):
    """
    One online cognition + learning step.
    """

    # 1. Raw experience enters buffer (no interpretation yet)
    buffer.add(observation)

    # 2. Assemble reasoning context
    reasoning_context = {
        "recent_buffer": buffer.get(),
        "belief_state": state.get_state(),
        "long_term_memory": memory.get_memory()[-10:]  # compressed view
    }

    # 3. LLM reasons (but cannot mutate anything)
    prompt = f"""
You are a frozen reasoning engine.

Context:
{json.dumps(reasoning_context, indent=2)}

Tasks:
1. Decide if any NEW EXPERIENCE is worth remembering.
2. Decide if any BELIEF needs refinement.
3. Propose NO MORE THAN:
   - one memory proposal
   - one state_delta

Output JSON ONLY in this format:

{{
  "memory_proposal": {{
    "entry": {{ ... }},
    "confidence": 0.0
  }} | null,

  "state_delta": {{
    "delta": {{ ... }},
    "confidence": 0.0
  }} | null
}}
"""

    llm_output = await llm_call(prompt)

    # 4. Memory proposal (experience-level learning)
    mem_prop = llm_output.get("memory_proposal")
    # Weak learning → memory
    if (
        mem_prop
        and mem_prop.get("confidence", 0) >= 0.25
        and mem_prop.get("entry")
    ):
        memory.commit(mem_prop)


    # 5. State proposal (belief-level learning)
    state_prop = llm_output.get("state_delta")
    if (
        state_prop
        and state_prop.get("confidence", 0) >= 0.6
        and state_prop.get("delta")
    ):
        await state.update_state(state_prop["delta"])

    # 6. Clear buffer (attention moves on)
    buffer.clear()


# --- Example driver ---
async def main():
    observations = [
    {"event": "api_call_failed", "reason": "timeout"},
    {"event": "api_call_failed", "reason": "timeout"},
    {"event": "api_call_failed", "reason": "timeout"},
    {"event": "retry_succeeded"},
    {"event": "api_call_failed", "reason": "timeout"},
]


    for obs in observations:
        await cognitive_step(obs)

    print("Final State:")
    print(json.dumps(state.get_state(), indent=2))

    print("\nPersistent Memory:")
    print(json.dumps(memory.get_memory(), indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

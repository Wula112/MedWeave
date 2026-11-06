"""
Shared State definitions for the LangGraph multi-agent system.
All state types used across agents are defined here.
"""
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class MainAgentState(TypedDict):
    """
    Global state shared between the Router and all Specialized Agents.
    This is the 'single source of truth' for the current conversation turn.
    """
    # Normally append-only; a successful compaction can replace the history.
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # A completed clinic subgraph hands its compacted history to the parent.
    history_replacement: Sequence[BaseMessage] | None

    # User profile info injected by the API layer (name, age, medical history, etc.)
    user_info: dict

    # The agent the Router decided to dispatch to, e.g., "clinic_agent", "insurance_agent"
    next_agent: str
    
    # Persistent tracking of the current focused agent to handle multi-turn conversations
    # If set, the router should bypass standard classification and route back here.
    active_agent: str

    # How many Command-based handoffs have already been honoured this turn.
    # Reset to 0 by router_node and capped by handoff.MAX_HANDOFFS_PER_TURN so
    # two specialists cannot transfer the same message back and forth forever.
    handoff_count: int

    # Which I/O channel produced this turn: "text" (default, /api/chat) or
    # "voice" (/api/voice). Only the clinic subgraph reads it, to add the
    # spoken read-back confirmation; text-channel behaviour is unchanged.
    channel: str

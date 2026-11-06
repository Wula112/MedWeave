"""Observation only: turn LangGraph events into the existing transport contract."""
from __future__ import annotations

import json
from typing import Any


def split_stream_chunk(chunk: Any) -> tuple[str, Any] | None:
    if not isinstance(chunk, tuple):
        return None
    if len(chunk) == 3:
        return chunk[1], chunk[2]
    if len(chunk) == 2 and isinstance(chunk[0], str):
        return chunk[0], chunk[1]
    return None


def interrupt_question(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("question", "content", "text"):
            if isinstance(value.get(key), str):
                return value[key]
    return json.dumps(value, ensure_ascii=False, default=str)


async def iter_graph_events(graph, graph_input, config, *, node_labels: dict,
                            tool_labels: dict, custom_types: set | frozenset):
    seen_interrupts: set[str] = set()
    stream = graph.astream_events(graph_input, config=config, version="v2",
                                 stream_mode=["updates", "custom"], subgraphs=True)
    try:
        async for event in stream:
            kind, name = event["event"], event.get("name", "")
            if kind == "on_chat_model_stream":
                if event.get("metadata", {}).get("lc_source") == "summarization":
                    continue
                content = event["data"]["chunk"].content
                if isinstance(content, list):
                    content = "".join(block["text"] for block in content
                        if isinstance(block, dict) and block.get("type") in ("text", "output_text")
                        and isinstance(block.get("text"), str))
                if isinstance(content, str) and content:
                    yield {"type": "text", "content": content}
            elif kind == "on_chain_stream":
                decoded = split_stream_chunk(event.get("data", {}).get("chunk"))
                if decoded is None:
                    continue
                mode, payload = decoded
                if mode == "custom" and isinstance(payload, dict) and payload.get("type") in custom_types:
                    yield payload
                elif mode == "updates" and isinstance(payload, dict):
                    for item in payload.get("__interrupt__", ()) or ():
                        marker = str(getattr(item, "id", "") or id(item))
                        if marker in seen_interrupts:
                            continue
                        seen_interrupts.add(marker)
                        value = getattr(item, "value", item)
                        question = interrupt_question(value)
                        yield {"type": "text", "content": question}
                        output = {"type": "interrupt", "content": question}
                        if isinstance(value, dict) and isinstance(value.get("kind"), str):
                            output["kind"] = value["kind"]
                        yield output
            elif kind == "on_chain_start" and name in node_labels:
                yield {"type": "node_start", "node": name, "content": node_labels[name]}
            elif kind == "on_chain_end" and name in node_labels:
                yield {"type": "node_end", "node": name}
            elif kind == "on_tool_start":
                yield {"type": "tool_start", "tool": name, "content": tool_labels.get(name, f"正在调用：{name}")}
            elif kind == "on_tool_end":
                yield {"type": "tool_end", "tool": name}
        yield {"type": "finish"}
    finally:
        await stream.aclose()

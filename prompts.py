"""prompts.py
Centralised prompt templates and rule helpers for the MKT3434 multi-agent RAG baseline.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

DEFAULT_RULES: dict[str, Any] = {
    "citation_style": "inline source tags",
    "writing_style": "clear technical",
    "output_structure": [],
    "max_response_length": 700,
    "domain_constraint": "",
    "language": "English",
}


def normalize_rules(rules: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_RULES)
    if isinstance(rules, dict):
        for key, value in rules.items():
            if value not in (None, "", []):
                merged[key] = value
    return merged


def format_rules_for_prompt(rules: dict[str, Any] | None) -> str:
    active = normalize_rules(rules)
    lines = []
    for key, value in active.items():
        if value in (None, "", []):
            continue
        pretty_key = key.replace("_", " ").title()
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {pretty_key}: {value}")
    return "\n".join(lines) if lines else "- No active rules."


def build_synthesizer_system_prompt(rules: dict[str, Any] | None) -> str:
    return (
        "You are the Synthesizer Agent in a multi-agent retrieval-augmented system. "
        "Write a grounded answer using only the provided evidence and tool outputs. "
        "Do not invent facts that are not supported by the evidence. "
        "When you use a source chunk, cite it inline using its chunk tag such as [C1] or [C2].\n\n"
        f"Active rules:\n{format_rules_for_prompt(rules)}"
    )



def build_synthesizer_user_prompt(
    query: str,
    sub_queries: Iterable[str],
    retrieved_chunks: list[dict[str, Any]],
    tool_context: dict[str, Any] | None,
) -> str:
    chunk_lines: list[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source_name") or chunk.get("source") or "unknown"
        page = chunk.get("page")
        page_suffix = f", page {page}" if page not in (None, "") else ""
        score = chunk.get("score")
        score_suffix = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
        text = chunk.get("text", "").strip()
        chunk_lines.append(
            f"[C{idx}] Source: {source}{page_suffix}{score_suffix}\n{text}"
        )

    tool_block = "No tool output."
    if tool_context:
        tool_block = json.dumps(tool_context, indent=2, ensure_ascii=False)

    sub_query_text = "\n".join(f"- {q}" for q in sub_queries) or f"- {query}"
    evidence = "\n\n".join(chunk_lines) if chunk_lines else "No retrieved chunks."

    return (
        f"Original user query:\n{query}\n\n"
        f"Sub-queries to cover:\n{sub_query_text}\n\n"
        f"Tool context:\n{tool_block}\n\n"
        f"Evidence chunks:\n{evidence}\n\n"
        "Write a concise but complete answer. Prefer direct synthesis over repetition. "
        "If evidence is insufficient, state that clearly and explain what is missing."
    )



def build_critic_prompt(
    query: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    rules: dict[str, Any] | None,
) -> str:
    evidence = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        evidence.append(f"[C{idx}] {chunk.get('text', '').strip()}")

    evidence_str = '\n\n'.join(evidence) if evidence else 'No evidence provided.'
    return (
        "You are the Critic Agent. Evaluate whether the answer is faithful to the evidence, "
        "complete with respect to the query, and aligned with the active rules.\n\n"
        f"Rules:\n{format_rules_for_prompt(rules)}\n\n"
        f"Query:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Evidence:\n{evidence_str}\n\n"
        "Return a JSON object with keys: faithfulness, completeness, overall_score, hallucinations, notes."
    )



def build_tool_decision_prompt(query: str, available_tools: list[dict[str, Any]]) -> str:
    tools = [
        {
            "name": tool.get("name", "unknown"),
            "description": tool.get("description", ""),
            "schema": tool.get("input_schema", {}),
        }
        for tool in available_tools
    ]
    return (
        "You are the Tool Agent. Decide whether a tool is necessary.\n\n"
        f"Query: {query}\n\n"
        f"Available tools:\n{json.dumps(tools, indent=2, ensure_ascii=False)}\n\n"
        "Return JSON with keys: use_tool (bool), tool_name, arguments, rationale."
    )

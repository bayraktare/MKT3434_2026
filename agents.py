"""agents.py
Baseline multi-agent implementation for the MKT3434 term project.
Students can extend each agent independently during the semester.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from prompts import (
    build_synthesizer_system_prompt,
    build_synthesizer_user_prompt,
    normalize_rules,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "the", "to", "what", "when", "where", "which",
    "who", "why", "with", "this", "that", "these", "those", "can", "could", "should",
    "would", "do", "does", "did", "into", "using", "use", "about", "explain", "summarise",
    "summarize", "compare", "between", "give", "write", "generate", "please", "me", "my",
}


@dataclass
class BaseAgent:
    name: str

    def trace(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"agent": self.name, **payload}


@dataclass
class MemoryReflectionAgent(BaseAgent):
    memory: list[dict[str, Any]] = field(default_factory=list)
    max_items: int = 12

    def remember(self, query: str, answer: str, metadata: dict[str, Any]) -> None:
        self.memory.append(
            {
                "query": query,
                "answer": answer,
                "score": metadata.get("overall_score"),
                "domain": metadata.get("used_domain"),
            }
        )
        if len(self.memory) > self.max_items:
            self.memory = self.memory[-self.max_items :]

    def retrieve_related(self, query: str) -> list[dict[str, Any]]:
        q_tokens = set(_keywords(query))
        scored = []
        for item in self.memory:
            overlap = len(q_tokens.intersection(_keywords(item["query"])))
            if overlap > 0:
                scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:2]]

    def reflect(self, critic_report: dict[str, Any], prior_domain: str, current_top_k: int) -> dict[str, Any]:
        suggestions = []
        if critic_report.get("faithfulness", 0.0) < 0.55:
            suggestions.append("Increase retrieved evidence and prioritize higher-overlap chunks.")
        if critic_report.get("completeness", 0.0) < 0.55:
            suggestions.append("Broaden the search scope and ensure each sub-query is addressed explicitly.")
        if prior_domain not in ("all", "general"):
            suggestions.append("Consider relaxing the domain filter on the next retry.")
        if not suggestions:
            suggestions.append("Keep the same domain but retrieve more context.")
        return self.trace(
            {
                "action": "reflection",
                "suggestions": suggestions,
                "next_top_k": current_top_k + 2,
            }
        )


@dataclass
class QueryReformulatorAgent(BaseAgent):
    def run(self, query: str, rules: dict[str, Any] | None = None) -> dict[str, Any]:
        cleaned = re.sub(r"\s+", " ", query).strip()
        parts = _split_query(cleaned)
        intent = _detect_intent(cleaned)
        complexity = "complex" if len(parts) > 1 or len(cleaned) > 140 else "simple"
        needs_tool = _looks_like_calculation(cleaned)
        return self.trace(
            {
                "rewritten_query": cleaned,
                "sub_queries": parts,
                "intent": intent,
                "complexity": complexity,
                "needs_tool": needs_tool,
            }
        )


@dataclass
class DomainRouterAgent(BaseAgent):
    domain_keywords: dict[str, list[str]] = field(
        default_factory=lambda: {
            "engineering": ["control", "robot", "mechanics", "sensor", "circuit", "machine", "engineering"],
            "health": ["health", "clinical", "disease", "patient", "medical", "exposure"],
            "sports": ["sport", "match", "player", "league", "team", "score"],
            "politics": ["policy", "government", "election", "minister", "parliament", "politics"],
            "general": [],
        }
    )

    def run(
        self,
        query: str,
        available_domains: list[str],
        rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_rules = normalize_rules(rules)
        forced_domain = str(active_rules.get("domain_constraint", "")).strip().lower()
        available = [d.lower() for d in available_domains] or ["all"]

        if forced_domain and forced_domain in available:
            return self.trace(
                {
                    "selected_domain": forced_domain,
                    "scores": {forced_domain: 1.0},
                    "reason": "Applied explicit domain constraint from rules.",
                }
            )

        scores: dict[str, float] = {domain: 0.0 for domain in available}
        q_tokens = _keywords(query)
        for domain in available:
            if domain in self.domain_keywords:
                scores[domain] += sum(1 for kw in self.domain_keywords[domain] if kw in q_tokens)
            if domain in q_tokens:
                scores[domain] += 2.0

        selected = max(scores.items(), key=lambda item: item[1])[0] if scores else "all"
        if scores.get(selected, 0.0) <= 0.0:
            selected = "all" if "all" in available else (available[0] if available else "all")

        return self.trace(
            {
                "selected_domain": selected,
                "scores": scores,
                "reason": "Keyword-based routing over available domains.",
            }
        )


@dataclass
class RetrieverAgent(BaseAgent):
    def run(
        self,
        plan: dict[str, Any],
        rag_pipeline: Any,
        domain: str = "all",
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> dict[str, Any]:
        collected: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        queries = plan.get("sub_queries") or [plan.get("rewritten_query") or ""]

        for query in queries:
            hits = rag_pipeline.retrieve(
                query=query,
                domain=domain,
                top_k=top_k,
                use_rerank=use_rerank,
            )
            for hit in hits:
                cid = hit.get("chunk_id")
                prev = seen.get(cid)
                if prev is None or hit.get("score", 0.0) > prev.get("score", 0.0):
                    seen[cid] = hit

        collected = sorted(seen.values(), key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
        return self.trace(
            {
                "queries_run": queries,
                "num_chunks": len(collected),
                "domain": domain,
                "chunks": collected,
            }
        )


@dataclass
class ToolMCPAgent(BaseAgent):
    def run(self, query: str, mcp_handler: Any | None = None) -> dict[str, Any]:
        if not _looks_like_calculation(query) and not _looks_like_file_read(query):
            return self.trace({"used_tool": False, "tool_name": None, "tool_output": None})

        tool_name = None
        arguments: dict[str, Any] = {}
        output: Any = None

        if _looks_like_calculation(query):
            tool_name = "calculator"
            arguments = {"expression": _extract_expression(query)}
        elif _looks_like_file_read(query):
            tool_name = "file_reader"
            arguments = {"path": _extract_path(query)}

        if mcp_handler is not None:
            output = mcp_handler.call_tool(tool_name, arguments)

        return self.trace(
            {
                "used_tool": True,
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_output": output,
            }
        )


@dataclass
class SynthesizerAgent(BaseAgent):
    def run(
        self,
        query: str,
        plan: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
        tool_context: dict[str, Any] | None,
        rules: dict[str, Any] | None,
        llm_callable: Callable[..., tuple[str, dict[str, Any]]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not retrieved_chunks and not tool_context:
            return (
                "I could not find enough evidence in the indexed documents to answer this query reliably.",
                {"used_llm": False, "fallback": True, "total_tokens": 0},
            )

        system_prompt = build_synthesizer_system_prompt(rules)
        user_prompt = build_synthesizer_user_prompt(
            query=query,
            sub_queries=plan.get("sub_queries") or [query],
            retrieved_chunks=retrieved_chunks,
            tool_context=tool_context,
        )

        if llm_callable is not None:
            try:
                answer, meta = llm_callable(system_prompt=system_prompt, user_prompt=user_prompt)
                meta = dict(meta or {})
                meta["used_llm"] = True
                return answer.strip(), meta
            except Exception:
                pass

        return self._fallback_answer(query, plan, retrieved_chunks, tool_context), {
            "used_llm": False,
            "fallback": True,
            "total_tokens": 0,
        }

    def _fallback_answer(
        self,
        query: str,
        plan: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
        tool_context: dict[str, Any] | None,
    ) -> str:
        lines = []
        if tool_context and tool_context.get("tool_output"):
            lines.append(f"Tool result: {json.dumps(tool_context['tool_output'], ensure_ascii=False)}")

        for idx, chunk in enumerate(retrieved_chunks[:3], start=1):
            excerpt = _first_sentences(chunk.get("text", ""), 2)
            source = chunk.get("source_name") or chunk.get("source") or "unknown source"
            lines.append(f"{excerpt} [C{idx}; {source}]")

        if not lines:
            return "I could not assemble a grounded answer from the available evidence."

        intro = "Grounded baseline answer:" if len(plan.get("sub_queries", [])) <= 1 else "Grounded baseline answer covering the detected sub-queries:"
        return intro + "\n\n" + "\n".join(f"- {line}" for line in lines)


@dataclass
class CriticAgent(BaseAgent):
    def run(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
        plan: dict[str, Any],
        rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chunk_texts = [chunk.get("text", "") for chunk in retrieved_chunks]
        answer_sentences = _split_sentences(answer)
        sentence_support = []
        hallucinations = []

        for sentence in answer_sentences:
            if len(_keywords(sentence)) < 3:
                continue
            best = max((_jaccard(sentence, chunk) for chunk in chunk_texts), default=0.0)
            sentence_support.append(best)
            if best < 0.08:
                hallucinations.append(sentence[:180])

        faithfulness = round(sum(sentence_support) / len(sentence_support), 3) if sentence_support else 0.4
        completeness = round(self._estimate_completeness(answer, plan), 3)

        active_rules = normalize_rules(rules)
        if active_rules.get("citation_style") and "[C" not in answer:
            faithfulness = round(max(0.0, faithfulness - 0.08), 3)

        overall = round((0.65 * faithfulness) + (0.35 * completeness), 3)
        return self.trace(
            {
                "faithfulness": faithfulness,
                "completeness": completeness,
                "overall_score": overall,
                "hallucinations": hallucinations[:5],
                "retry_recommended": overall < 0.62,
            }
        )

    def _estimate_completeness(self, answer: str, plan: dict[str, Any]) -> float:
        sub_queries = plan.get("sub_queries") or [plan.get("rewritten_query") or ""]
        if not sub_queries:
            return 0.5
        answer_tokens = set(_keywords(answer))
        covered = 0
        for sq in sub_queries:
            sq_tokens = [tok for tok in _keywords(sq) if tok not in STOPWORDS]
            if not sq_tokens:
                covered += 1
                continue
            if any(tok in answer_tokens for tok in sq_tokens[:6]):
                covered += 1
        return covered / max(1, len(sub_queries))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _split_query(query: str) -> list[str]:
    pieces = [p.strip(" -•\t") for p in re.split(r"\?|\n|;|\b(?:also|and also|in addition)\b", query) if p.strip()]
    if len(pieces) == 1 and " and " in query.lower() and len(query) > 80:
        pieces = [p.strip() for p in re.split(r"\band\b", query) if len(p.strip()) > 12]
    clean = []
    for item in pieces:
        if item and item not in clean:
            clean.append(item)
    return clean[:4] or [query]



def _detect_intent(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["compare", "difference", "versus", "vs"]):
        return "comparison"
    if any(word in q for word in ["summar", "overview", "state of the art"]):
        return "summary"
    if any(word in q for word in ["list", "bullet", "enumerate"]):
        return "listing"
    if any(word in q for word in ["write", "generate", "draft"]):
        return "generation"
    return "qa"



def _looks_like_calculation(query: str) -> bool:
    q = query.lower()
    return bool(re.search(r"\d+\s*[-+/*^()]\s*\d+", q)) or any(
        token in q for token in ["calculate", "compute", "solve", "evaluate", "convert"]
    )



def _extract_expression(query: str) -> str:
    math_like = re.findall(r"[\d\s\.+\-/*^()]+", query)
    candidates = [c.strip() for c in math_like if any(ch.isdigit() for ch in c) and len(c.strip()) >= 3]
    return candidates[0] if candidates else query



def _looks_like_file_read(query: str) -> bool:
    return bool(re.search(r"(?:read|open|inspect)\s+[^\n]+\.(?:txt|md|json|csv|html)", query.lower()))



def _extract_path(query: str) -> str:
    match = re.search(r"([\w./\\-]+\.(?:txt|md|json|csv|html))", query)
    return match.group(1) if match else ""



def _keywords(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if t not in STOPWORDS and len(t) > 1]



def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]



def _jaccard(a: str, b: str) -> float:
    sa = set(_keywords(a))
    sb = set(_keywords(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)



def _first_sentences(text: str, n: int = 2) -> str:
    sentences = _split_sentences(re.sub(r"\s+", " ", text.strip()))
    if not sentences:
        return text[:220].strip()
    return " ".join(sentences[:n]).strip()

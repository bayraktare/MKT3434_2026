"""orchestrator.py
Orchestrates the multi-agent RAG workflow behind the GUI-facing RAGPipeline interface.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from agents import (
    CriticAgent,
    DomainRouterAgent,
    MemoryReflectionAgent,
    QueryReformulatorAgent,
    RetrieverAgent,
    SynthesizerAgent,
    ToolMCPAgent,
)
from prompts import normalize_rules


class Orchestrator:
    """Coordinates the baseline agent workflow and retry loop."""

    def __init__(
        self,
        rag_pipeline: Any,
        mcp_handler: Any | None = None,
        quality_threshold: float = 0.62,
        max_retries: int = 1,
    ):
        self.rag_pipeline = rag_pipeline
        self.mcp_handler = mcp_handler
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries

        self.reformulator = QueryReformulatorAgent("reformulator")
        self.router = DomainRouterAgent("domain_router")
        self.tool_agent = ToolMCPAgent("tool_mcp")
        self.retriever = RetrieverAgent("retriever")
        self.synthesizer = SynthesizerAgent("synthesizer")
        self.critic = CriticAgent("critic")
        self.memory = MemoryReflectionAgent("memory_reflection")

    def run(
        self,
        query: str,
        model: str,
        rules: dict[str, Any] | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        active_rules = normalize_rules(rules)
        agent_trace: list[dict[str, Any]] = []
        token_counter = 0

        related_memory = self.memory.retrieve_related(query)
        if related_memory:
            agent_trace.append(
                {
                    "agent": "memory_reflection",
                    "action": "retrieve_related",
                    "items": related_memory,
                }
            )

        plan = self.reformulator.run(query, active_rules)
        agent_trace.append(plan)

        available_domains = self.rag_pipeline.available_domains()
        route = self.router.run(plan["rewritten_query"], available_domains, active_rules)
        agent_trace.append(route)

        tool_result = self.tool_agent.run(query, self.mcp_handler)
        agent_trace.append(tool_result)

        selected_domain = route.get("selected_domain", "all")
        top_k = self.rag_pipeline.top_k
        final_answer = ""
        final_synth_meta: dict[str, Any] = {}
        critic_report: dict[str, Any] = {}
        retrieval_result: dict[str, Any] = {"chunks": []}

        llm_callable = lambda system_prompt, user_prompt: self.rag_pipeline.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )

        for attempt in range(self.max_retries + 1):
            retrieval_result = self.retriever.run(
                plan=plan,
                rag_pipeline=self.rag_pipeline,
                domain=selected_domain,
                top_k=top_k,
                use_rerank=True,
            )
            agent_trace.append(
                {
                    "agent": "retriever",
                    "attempt": attempt,
                    "domain": selected_domain,
                    "top_k": top_k,
                    "num_chunks": retrieval_result.get("num_chunks", 0),
                    "queries_run": retrieval_result.get("queries_run", []),
                }
            )

            final_answer, final_synth_meta = self.synthesizer.run(
                query=query,
                plan=plan,
                retrieved_chunks=retrieval_result.get("chunks", []),
                tool_context=tool_result,
                rules=active_rules,
                llm_callable=llm_callable,
            )
            token_counter += int(final_synth_meta.get("total_tokens", 0) or 0)
            agent_trace.append(
                {
                    "agent": "synthesizer",
                    "attempt": attempt,
                    "used_llm": final_synth_meta.get("used_llm", False),
                    "fallback": final_synth_meta.get("fallback", False),
                }
            )

            critic_report = self.critic.run(
                query=query,
                answer=final_answer,
                retrieved_chunks=retrieval_result.get("chunks", []),
                plan=plan,
                rules=active_rules,
            )
            critic_report["attempt"] = attempt
            agent_trace.append(critic_report)

            if critic_report.get("overall_score", 0.0) >= self.quality_threshold:
                break
            if attempt >= self.max_retries:
                break

            reflection = self.memory.reflect(critic_report, selected_domain, top_k)
            agent_trace.append(reflection)
            top_k = int(reflection.get("next_top_k", top_k + 2))
            if selected_domain not in ("all", "general"):
                selected_domain = "all"

        self.rag_pipeline.set_last_retrieved_chunks(retrieval_result.get("chunks", []))

        if stream_callback:
            self._emit_response(final_answer, stream_callback)

        metadata = {
            "model": model,
            "total_tokens": token_counter,
            "agent_trace": agent_trace,
            "num_chunks_retrieved": len(retrieval_result.get("chunks", [])),
            "used_domain": selected_domain,
            "tool_used": bool(tool_result.get("used_tool")),
            "faithfulness": critic_report.get("faithfulness"),
            "completeness": critic_report.get("completeness"),
            "overall_score": critic_report.get("overall_score"),
            "hallucinations": critic_report.get("hallucinations", []),
            "retry_count": critic_report.get("attempt", 0),
            "elapsed_orchestration_sec": round(time.perf_counter() - t0, 3),
        }

        self.memory.remember(query, final_answer, metadata)
        return final_answer, metadata

    @staticmethod
    def _emit_response(text: str, callback: Callable[[str], None]) -> None:
        if not text:
            return
        words = text.split()
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            callback(word + suffix)

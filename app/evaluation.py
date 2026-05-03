import re
import time
from collections import Counter

from app.observability import log_evaluation, new_trace_id
from app.rag.pipeline import (
    RESPONSE_SCHEMA,
    call_gemini_llm,
    get_model_name,
    parse_llm_json,
    retrieve_rag_context,
)
from app.rag.prompt import build_prompt

PROMPT_VARIANTS = {
    "baseline": {
        "label": "Baseline Grounded JSON",
        "description": "Current strict RAG prompt with answer, insights, risks, recommendation, and sources.",
    },
    "concise": {
        "label": "Concise Decision Brief",
        "description": "Shorter decision-focused prompt emphasizing directness and minimal repetition.",
    },
    "risk_first": {
        "label": "Risk-First Decision Brief",
        "description": "Prompt variant that prioritizes risks, uncertainty, and recommendation quality.",
    },
}

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "you", "your", "have", "has", "had", "but", "not", "all", "can", "will",
    "its", "into", "about", "they", "their", "there", "what", "when", "where",
    "which", "would", "should", "could", "also", "than", "then", "them",
}


def get_prompt_variants() -> dict:
    return PROMPT_VARIANTS


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def _build_variant_prompt(variant: str, context_chunks: list[dict], question: str) -> str:
    base_prompt = build_prompt(context_chunks, question)

    if variant == "concise":
        return f"""
{base_prompt}

Additional prompt variant instructions:
- Prefer the shortest complete answer.
- Keep each key insight under 18 words.
- Include only the strongest risks and the most practical recommendation.
"""

    if variant == "risk_first":
        return f"""
{base_prompt}

Additional prompt variant instructions:
- Prioritize risk analysis before optimism.
- Clearly separate known facts from uncertainty.
- Make the recommendation cautious when evidence is limited.
"""

    return base_prompt


def _groundedness_metrics(response: dict, context_chunks: list[dict]) -> dict:
    answer_text = " ".join([
        str(response.get("answer", "")),
        " ".join(str(item) for item in response.get("key_insights", [])),
        " ".join(str(item) for item in response.get("risks", [])),
        str(response.get("recommendation", "")),
    ])
    context_text = " ".join(str(chunk.get("text", "")) for chunk in context_chunks)
    answer_tokens = _tokens(answer_text)
    context_tokens = set(_tokens(context_text))

    if not answer_tokens:
        support_ratio = 0.0
    else:
        supported = sum(1 for token in answer_tokens if token in context_tokens)
        support_ratio = supported / len(answer_tokens)

    source_pages = {
        chunk.get("page")
        for chunk in context_chunks
        if chunk.get("page") is not None
    }
    cited_pages = {
        source.get("page")
        for source in response.get("sources", [])
        if isinstance(source, dict) and source.get("page") is not None
    }
    valid_citations = cited_pages.issubset(source_pages) if cited_pages else False
    citation_coverage = 0.0
    if source_pages:
        citation_coverage = len(cited_pages.intersection(source_pages)) / len(source_pages)

    repeated_terms = Counter(answer_tokens).most_common(5)

    return {
        "groundedness_score": round(support_ratio, 3),
        "citation_coverage": round(citation_coverage, 3),
        "valid_citations": valid_citations,
        "answer_token_count": len(answer_tokens),
        "source_page_count": len(source_pages),
        "cited_page_count": len(cited_pages),
        "top_answer_terms": [
            {"term": term, "count": count}
            for term, count in repeated_terms
        ],
    }


def _quality_metrics(response: dict) -> dict:
    return {
        "has_answer": bool(str(response.get("answer", "")).strip()),
        "insight_count": len(response.get("key_insights", [])),
        "risk_count": len(response.get("risks", [])),
        "source_count": len(response.get("sources", [])),
        "has_recommendation": bool(str(response.get("recommendation", "")).strip()),
    }


def compare_prompts(question: str, variants: list[str] | None = None) -> dict:
    trace_id = new_trace_id()
    selected_variants = variants or list(PROMPT_VARIANTS.keys())
    selected_variants = [
        variant for variant in selected_variants
        if variant in PROMPT_VARIANTS
    ] or ["baseline"]

    retrieval_started = time.perf_counter()
    context_chunks, retrieval_metrics = retrieve_rag_context(question)
    retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)

    results = []
    for variant in selected_variants:
        prompt = _build_variant_prompt(variant, context_chunks, question)
        generation_started = time.perf_counter()
        raw_output = call_gemini_llm(
            prompt,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=2048,
        )
        generation_latency_ms = round((time.perf_counter() - generation_started) * 1000, 2)
        response = parse_llm_json(raw_output, context_chunks)
        groundedness = _groundedness_metrics(response, context_chunks)
        quality = _quality_metrics(response)
        overall_score = round(
            groundedness["groundedness_score"] * 0.5
            + groundedness["citation_coverage"] * 0.2
            + (1.0 if quality["has_recommendation"] else 0.0) * 0.1
            + min(quality["source_count"], 3) / 3 * 0.2,
            3,
        )

        results.append({
            "variant": variant,
            "label": PROMPT_VARIANTS[variant]["label"],
            "description": PROMPT_VARIANTS[variant]["description"],
            "answer": response,
            "metrics": {
                **groundedness,
                **quality,
                "overall_score": overall_score,
                "latency_ms": generation_latency_ms,
                "prompt_length": len(prompt),
                "response_length": len(raw_output),
            },
        })

    results = sorted(
        results,
        key=lambda item: (
            item["metrics"]["overall_score"],
            item["metrics"]["groundedness_score"],
            -item["metrics"]["latency_ms"],
        ),
        reverse=True,
    )

    run = {
        "event_type": "prompt_comparison",
        "trace_id": trace_id,
        "model": get_model_name(),
        "question": question,
        "retrieval": {
            **retrieval_metrics,
            "latency_ms": retrieval_latency_ms,
        },
        "winner": results[0]["variant"] if results else None,
        "results": results,
    }
    return log_evaluation(run)

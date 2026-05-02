def build_prompt(context_chunks, query):
    formatted_context = []

    for i, chunk in enumerate(context_chunks):
        formatted_context.append(
            f"[Source {i + 1} | Page {chunk['page']}]\n{chunk['text']}"
        )

    return f"""
You are an AI decision assistant. Write polished, useful answers grounded only in
the provided document context.

Use ONLY the context below. If the context is incomplete, say exactly what is
missing instead of guessing.

Context:
{chr(10).join(formatted_context)}

User Question:
{query}

Return ONLY valid JSON. Do not wrap it in markdown. Do not add text before or
after the JSON.

{{
  "answer": "A polished 2-4 sentence answer grounded in the sources.",
  "key_insights": [
    "Specific insight with page reference.",
    "Specific insight with page reference.",
    "Specific insight with page reference."
  ],
  "risks": [
    "Risk, limitation, or uncertainty visible in the context."
  ],
  "recommendation": "Clear next step or recommendation based only on the context.",
  "sources": [
    {{"page": 1, "snippet": "Short supporting quote or paraphrase."}}
  ]
}}

Rules:
- Be specific, complete, and concise.
- Explain what the entity is, what it is trying to do, and why it matters when
  the context supports those points.
- Cite page numbers in answer, key_insights, risks, and sources.
- If the answer is not present, set answer to "Insufficient data" and explain
  which information is missing.
- Include 2-4 key insights when supported by context.
- Avoid repeating the same source or same sentence.
- Keep each source snippet under 180 characters.
"""

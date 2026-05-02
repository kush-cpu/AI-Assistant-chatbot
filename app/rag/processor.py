import fitz  # PyMuPDF
import re


def _clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def _build_overlap(text, max_chars):
    sentences = _split_sentences(text)
    overlap_parts = []
    overlap_length = 0

    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        next_length = overlap_length + len(sentence) + 1
        if next_length > max_chars and overlap_parts:
            break

        overlap_parts.insert(0, sentence)
        overlap_length = next_length

    return " ".join(overlap_parts).strip()

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text_data = []

    for i, page in enumerate(doc): # type: ignore
        text = _clean_text(page.get_text())
        if not text:
            continue

        text_data.append({
            "page": i + 1,
            "content": text
        })

    return text_data


def chunk_text(text_data, chunk_size=900, overlap=180):
    chunks = []

    for item in text_data:
        content = item["content"]
        sentences = _split_sentences(content)
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - overlap):
                    chunk = sentence[i:i + chunk_size].strip()
                    if chunk:
                        chunks.append({
                            "text": chunk,
                            "page": item["page"]
                        })
                continue

            next_text = f"{current} {sentence}".strip()
            if len(next_text) <= chunk_size:
                current = next_text
                continue

            if current:
                chunks.append({
                    "text": current,
                    "page": item["page"]
                })

            current = _build_overlap(current, overlap)
            current = f"{current} {sentence}".strip()

        if current:
            chunks.append({
                "text": current,
                "page": item["page"]
            })

    return chunks

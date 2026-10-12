"""
RAG service for PrintGuard.
"""

import os
import re
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


BASE_DIR = Path(__file__).parent
CHROMA_PATH = os.environ.get("CHROMA_DB_PATH", str(BASE_DIR / "data" / "chroma_db"))
KB_PATH = os.environ.get("KNOWLEDGE_BASE_PATH", str(BASE_DIR / "data" / "knowledge_base.json"))

UNIFIED_CLASSES = ["spaghetti", "warping", "under_extrusion", "stringing"]
MODEL_NAME = "openai/gpt-oss-120b"
COLLECTION_NAME = "printguard_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"


_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY env var is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


_collection = None

def _init_collection():
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    try:
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
        print(f"[ai_service] Loaded ChromaDB collection: {_collection.count()} docs")
        return _collection
    except Exception:
        pass

    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"ChromaDB missing and KB not found at {KB_PATH}")

    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)

    _collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    _collection.add(
        ids=[d["doc_id"] for d in kb],
        documents=[d["text"] for d in kb],
        metadatas=[{
            "class":  d["class"],
            "title":  d["title"],
            "source": d["source"],
            "url":    d.get("url", ""),
        } for d in kb],
    )
    print(f"[ai_service] Built ChromaDB from KB: {_collection.count()} docs")
    return _collection


SYSTEM_PROMPT = """You are a 3D-print defect diagnostic assistant.

RULES:
1. Every factual claim MUST come from the provided CONTEXT.
2. Cite source doc IDs in brackets, e.g. [warping_014].
3. If CONTEXT does not cover it, respond "Not covered in the knowledge base."
4. Do NOT invent causes, temperatures, materials, or fixes.
5. Structure: Root cause / Suggested fix / Urgency.
6. Keep under 180 words.
"""

PROACTIVE_TEMPLATE = """DETECTED DEFECT: {defect_class} (confidence {confidence:.2f})

CONTEXT:
{context_block}

Provide a root cause analysis, a suggested fix, and an urgency level.
Cite each claim with [doc_id].
"""

CONVERSATIONAL_TEMPLATE = """DETECTED DEFECT: {defect_class} (confidence {confidence:.2f})

CONTEXT:
{context_block}

USER QUESTION: {user_question}

Answer using ONLY the CONTEXT. Cite each claim with [doc_id].
"""

CITATION_RE = re.compile(r"[\[【［]\s*(\w+_\d+)\s*[\]】］]")


def retrieve_for_defect(defect_class, k=3):
    collection = _init_collection()
    results = collection.query(
        query_texts=[f"causes and fixes for {defect_class} in FDM 3D printing"],
        n_results=k,
        where={"class": defect_class},
    )
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "doc_id":   results["ids"][0][i],
            "text":     results["documents"][0][i],
            "title":    results["metadatas"][0][i]["title"],
            "source":   results["metadatas"][0][i]["source"],
            "url":      results["metadatas"][0][i].get("url", ""),
            "distance": results["distances"][0][i],
        })
    return docs


def _build_context(docs):
    return "\n\n".join(f"[{d['doc_id']}] {d['title']}\n{d['text']}" for d in docs)


def _extract_content(resp):
    msg = resp.choices[0].message
    content = (getattr(msg, "content", None) or "").strip()
    if content:
        return content
    reasoning = (getattr(msg, "reasoning", None) or "").strip()
    return reasoning if reasoning else ""


def _call_groq(system_prompt, user_prompt, max_tokens=1200):
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return _extract_content(resp)


def _validate_citations(response, expected_ids):
    cited = set(CITATION_RE.findall(response))
    invalid = cited - set(expected_ids)
    if invalid:
        return False, f"Invalid citations: {invalid}"
    if not cited:
        return False, "No citations found"
    return True, "OK"


def _normalize_citations(response):
    return CITATION_RE.sub(r"[\1]", response)


def diagnose(defect_class, confidence, user_question=None):
    if defect_class not in UNIFIED_CLASSES:
        return {
            "analysis": f"Unknown defect class '{defect_class}'.",
            "citations": [],
            "validation": "unknown_class",
            "sources": [],
        }

    docs = retrieve_for_defect(defect_class, k=3)
    if not docs:
        return {
            "analysis": f"No knowledge base entries for '{defect_class}'.",
            "citations": [],
            "validation": "no_docs",
            "sources": [],
        }

    context_block = _build_context(docs)

    if user_question:
        user_prompt = CONVERSATIONAL_TEMPLATE.format(
            defect_class=defect_class,
            confidence=confidence,
            context_block=context_block,
            user_question=user_question,
        )
    else:
        user_prompt = PROACTIVE_TEMPLATE.format(
            defect_class=defect_class,
            confidence=confidence,
            context_block=context_block,
        )

    try:
        response = _call_groq(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return {
            "analysis": f"LLM call failed: {e}",
            "citations": [d["doc_id"] for d in docs],
            "validation": "llm_error",
            "sources": docs,
        }

    doc_ids = [d["doc_id"] for d in docs]
    ok, msg = _validate_citations(response, doc_ids)

    if not ok:
        strict_system = SYSTEM_PROMPT + "\n7. You MUST cite at least one [doc_id] in EVERY bullet point."
        try:
            response = _call_groq(strict_system, user_prompt)
            ok, msg = _validate_citations(response, doc_ids)
        except Exception:
            pass

    if not ok:
        try:
            response = _call_groq(SYSTEM_PROMPT, user_prompt)
            ok, msg = _validate_citations(response, doc_ids)
        except Exception:
            pass

    if not ok:
        causes = "\n".join(f"- [{d['doc_id']}] {d['title']}" for d in docs)
        response = (
            f"Detected {defect_class} with confidence {confidence:.2f}.\n\n"
            f"Relevant references from knowledge base:\n{causes}\n\n"
            f"(Automated explanation unavailable — validation failed: {msg})"
        )
        return {
            "analysis": response,
            "citations": doc_ids,
            "validation": msg,
            "sources": docs,
        }

    return {
        "analysis": _normalize_citations(response),
        "citations": doc_ids,
        "validation": "OK",
        "sources": [{
            "doc_id": d["doc_id"],
            "title":  d["title"],
            "source": d["source"],
            "url":    d["url"],
        } for d in docs],
    }


if __name__ == "__main__":
    print("Testing RAG service...")
    result = diagnose("warping", 0.82)
    print(json.dumps(result, indent=2, default=str))
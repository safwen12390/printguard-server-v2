"""
ONNX-based RAG service — parallel version for memory-constrained deployment.
Does NOT touch ai_service.py or data/chroma_db/.
Uses a separate ChromaDB folder (data/chroma_db_onnx) so both can coexist.
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # <-- add this

BASE_DIR = Path(__file__).parent
CHROMA_PATH = os.environ.get("CHROMA_DB_ONNX_PATH", str(BASE_DIR / "data" / "chroma_db_onnx"))
KB_PATH = os.environ.get("KNOWLEDGE_BASE_PATH", str(BASE_DIR / "data" / "knowledge_base.json"))
ONNX_DIR = os.environ.get("ONNX_MODEL_DIR", str(BASE_DIR / "onnx_miniLM"))

UNIFIED_CLASSES = ["spaghetti", "warping", "under_extrusion", "stringing"]
MODEL_NAME = "openai/gpt-oss-120b"
COLLECTION_NAME = "printguard_kb"   # same collection name, different DB folder

_groq_client = None
_collection = None
_onnx_session = None
_onnx_tokenizer = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY env var is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_onnx_embedder():
    global _onnx_session, _onnx_tokenizer
    if _onnx_session is not None and _onnx_tokenizer is not None:
        return _onnx_session, _onnx_tokenizer

    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_path = os.path.join(ONNX_DIR, "model.onnx")
    tokenizer_path = os.path.join(ONNX_DIR, "tokenizer.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _onnx_session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    _onnx_tokenizer = Tokenizer.from_file(tokenizer_path)
    _onnx_tokenizer.enable_truncation(max_length=128)
    return _onnx_session, _onnx_tokenizer


class ONNXEmbedder:
    def __init__(self):
        self._name = "onnx-minilm"

    def name(self):
        # ChromaDB checks this to identify the embedder for persistence
        return self._name

    def __call__(self, input):
        import numpy as np

        session, tokenizer = _get_onnx_embedder()

        if isinstance(input, str):
            texts = [input]
            single = True
        else:
            texts = list(input)
            single = False

        encodings = tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encodings)

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            n = len(enc.ids)
            input_ids[i, :n] = enc.ids
            attention_mask[i, :n] = enc.attention_mask

        outputs = session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden = outputs[0]

        mask_f = attention_mask[..., None].astype(np.float32)
        summed = (last_hidden * mask_f).sum(axis=1)
        counts = mask_f.sum(axis=1)
        counts = np.clip(counts, a_min=1e-9, a_max=None)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        normalized = pooled / norms

        result = normalized.tolist()
        return result[0] if single else result

    # ChromaDB calls these for query-time and index-time embedding
    def embed_query(self, input):
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

_embedder_singleton = None

def _get_embedder():
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = ONNXEmbedder()
    return _embedder_singleton


def _init_collection():
    global _collection
    if _collection is not None:
        return _collection

    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = _get_embedder()

    try:
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
        print(f"[ai_service_onnx] Loaded ChromaDB: {_collection.count()} docs")
        return _collection
    except Exception as e:
        print(f"[ai_service_onnx] get_collection failed: {type(e).__name__}: {e}")

    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"KB not found at {KB_PATH}")

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
    print(f"[ai_service_onnx] Built ChromaDB from KB: {_collection.count()} docs")
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

CITATION_RE = re.compile(r"[\[\u3010\uff3b]\s*(\w+_\d+)\s*[\]\u3011\uff3d]")


def retrieve_for_defect(defect_class, k=3, user_question=None):
    collection = _init_collection()
    if user_question and user_question.strip():
        query = f"{user_question.strip()} (about {defect_class} in FDM 3D printing)"
    else:
        query = f"causes and fixes for {defect_class} in FDM 3D printing"
    results = collection.query(
        query_texts=[query],
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
        return {"analysis": f"Unknown defect class '{defect_class}'.",
                "citations": [], "validation": "unknown_class", "sources": []}

    docs = retrieve_for_defect(defect_class, k=3, user_question=user_question)
    if not docs:
        return {"analysis": f"No knowledge base entries for '{defect_class}'.",
                "citations": [], "validation": "no_docs", "sources": []}

    context_block = _build_context(docs)

    if user_question:
        user_prompt = CONVERSATIONAL_TEMPLATE.format(
            defect_class=defect_class, confidence=confidence,
            context_block=context_block, user_question=user_question)
    else:
        user_prompt = PROACTIVE_TEMPLATE.format(
            defect_class=defect_class, confidence=confidence,
            context_block=context_block)

    try:
        response = _call_groq(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return {"analysis": f"LLM call failed: {e}",
                "citations": [d["doc_id"] for d in docs],
                "validation": "llm_error", "sources": docs}

    doc_ids = [d["doc_id"] for d in docs]
    ok, msg = _validate_citations(response, doc_ids)

    if not ok:
        strict = SYSTEM_PROMPT + "\n7. You MUST cite at least one [doc_id] in EVERY bullet point."
        try:
            response = _call_groq(strict, user_prompt)
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
        response = (f"Detected {defect_class} with confidence {confidence:.2f}.\n\n"
                    f"Relevant references:\n{causes}\n\n"
                    f"(Automated explanation unavailable - validation: {msg})")
        return {"analysis": response, "citations": doc_ids,
                "validation": msg, "sources": docs}

    return {
        "analysis": _normalize_citations(response),
        "citations": doc_ids,
        "validation": "OK",
        "sources": [{"doc_id": d["doc_id"], "title": d["title"],
                     "source": d["source"], "url": d["url"]} for d in docs],
    }


if __name__ == "__main__":
    print("Testing ONNX RAG service...")
    result = diagnose("warping", 0.82)
    print(json.dumps(result, indent=2, default=str))
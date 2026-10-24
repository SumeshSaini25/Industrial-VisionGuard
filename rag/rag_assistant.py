
from pathlib import Path
import argparse
import json
import sys
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.search_knowledge_base import search_knowledge_base


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:4b"


def ask_assistant(question):
    matches = search_knowledge_base(question, top_k=3)

    if not matches:
        raise RuntimeError("No knowledge-base passages found.")

    context_parts = []

    for index, match in enumerate(matches, start=1):
        context_parts.append(
            f"[Source {index}: {match['source']}]\n"
            f"{match['content']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are Industrial VisionGuard's quality-control assistant.

Use the supplied reference documents for quality-control guidance.
Use the inspection facts supplied in the question only as
reported model outputs, not as independently verified measurements.

CRITICAL RULES:
- PatchCore anomaly scores are NOT probabilities or confidence levels.
- NEVER convert an anomaly score into a percentage.
- Preserve the original model prediction: GOOD or DEFECT.
- Do not identify a specific defect type from PatchCore alone.
- Do not invent causes, severity, acceptance limits, or standards.
- Distinguish possible causes from confirmed observations.
- Recommend human verification against approved specifications.
- If evidence is insufficient, explicitly state the limitation.
- Cite retrieved source numbers for document-based guidance.
- Do not cite a document as proof of a model-generated observation.
- Do not reveal internal reasoning or analysis.

Write a concise, professional inspection report.
"""

    user_prompt = f"""
REFERENCE DOCUMENTS:
{context}

QUESTION:
{question}

Provide an evidence-grounded answer with source citations.
"""

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.2},
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Cannot reach Ollama. Start the Ollama application "
            "and verify that qwen3:4b is installed."
        ) from error

    answer = result["message"].get("content", "").strip()

    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    if not answer:
        raise RuntimeError(
            "Qwen returned no final answer."
        )

    return {
        "question": question,
        "answer": answer,
        "sources": matches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ask the local Industrial VisionGuard RAG assistant"
    )
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    print("\nSearching knowledge base and asking Qwen3...\n")

    result = ask_assistant(args.query)

    print("=" * 60)
    print("INDUSTRIAL VISIONGUARD - RAG ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\nRetrieved sources:")
    for index, source in enumerate(result["sources"], start=1):
        print(
            f"[{index}] {source['source']} "
            f"(distance: {source['distance']:.4f})"
        )


if __name__ == "__main__":
    main()

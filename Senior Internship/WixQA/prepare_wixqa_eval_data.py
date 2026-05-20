from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

#Reading jsonl 
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc

#Loading documents 
def load_kb_lookup(kb_path: Path, max_docs : int = None) -> Dict[str, str]:
    kb_lookup: Dict[str, str] = {}
    cnt = 0
    for row in iter_jsonl(kb_path):
        doc_id = row.get("id")
        contents = row.get("contents")
        if not doc_id or not isinstance(contents, str):
            continue
        kb_lookup[doc_id] = contents
        cnt+=1
        if max_docs != None and cnt > max_docs: 
            break
    return kb_lookup

#Build dataset.json 
def build_eval_rows(
    qa_path: Path,
    kb_lookup: Dict[str, str],
    max_samples: int | None = None,
    max_chars_per_context_doc: int | None = None,
) -> List[dict]:
    rows: List[dict] = []
    missing_contexts = 0

    for idx, row in enumerate(iter_jsonl(qa_path), start=1):
        question = row.get("question", "")
        answer = row.get("answer", "")
        article_ids = row.get("article_ids", [])

        if not question or not answer:
            continue

        if not isinstance(article_ids, list):
            article_ids = []

        context_docs: List[str] = []
        for article_id in article_ids:
            content = kb_lookup.get(article_id)
            if content is None:
                continue
            if max_chars_per_context_doc is not None:
                content = content[:max_chars_per_context_doc]
            context_docs.append(content)

        if not context_docs:
            missing_contexts += 1

        rows.append(
            {
                "id": f"wixqa-{idx}",
                "input": question,
                "actual_output": None,
                "expected_output": answer,
                "context": context_docs,
                "source_file": str(qa_path),
                "article_ids": article_ids,
            }
        )

        if max_samples is not None and len(rows) >= max_samples:
            break

    print(f"Prepared {len(rows)} rows from {qa_path.name}")
    print(f"Rows with zero gold context docs: {missing_contexts}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DeepEval-compatible JSON dataset from local WixQA JSONL files."
        )
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("./test.jsonl"),
        help="Path to WixQA QA JSONL file (default: ./test.jsonl)",
    )
    parser.add_argument(
        "--kb",
        type=Path,
        default=Path("./wix_kb_corpus.jsonl"),
        help="Path to WixQA KB corpus JSONL file (default: ./wix_kb_corpus.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./wixqa_eval_dataset.json"),
        help="Output JSON path for DeepEval rows",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for number of rows to export",
    )
    parser.add_argument(
        "--max-chars-per-context-doc",
        type=int,
        default=None,
        help="Optional max chars to keep per gold context doc",
    )

    args = parser.parse_args()

    if not args.qa.exists():
        raise FileNotFoundError(f"QA file not found: {args.qa}")
    if not args.kb.exists():
        raise FileNotFoundError(f"KB file not found: {args.kb}")

    kb_lookup = load_kb_lookup(args.kb)
    print(f"Loaded {len(kb_lookup)} knowledge-base documents")

    rows = build_eval_rows(
        qa_path=args.qa,
        kb_lookup=kb_lookup,
        max_samples=args.max_samples,
        max_chars_per_context_doc=args.max_chars_per_context_doc,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset to {args.out}")


if __name__ == "__main__":
    main()

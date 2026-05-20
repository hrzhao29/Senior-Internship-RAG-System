
WixQA -> DeepEval pipeline (local files)

Use this when you want to evaluate on the local `WixQA` folder (`test.jsonl` + `wix_kb_corpus.jsonl`).

1. Build a DeepEval-compatible dataset with gold context docs:

`python prepare_wixqa_eval_data.py --out docs/wixqa_eval_dataset.json`

Optional flags:
- `--max-samples 100` to run a smaller subset first
- `--max-chars-per-context-doc 6000` to reduce token usage

What this script does:
- Reads `WixQA/test.jsonl` for `question`, `answer`, and `article_ids`
- Reads `WixQA/wix_kb_corpus.jsonl` for `id` -> `contents`
- Builds rows in the format expected by `test_eval.py`:
	- `input`
	- `expected_output`
	- `context` (list of full gold docs from `article_ids`)
	- `article_ids`

2. Run your existing DeepEval pipeline on the converted dataset:

`python test_eval.py --dataset docs/wixqa_eval_dataset.json`

Alternative:

`DATASET_PATH=docs/wixqa_eval_dataset.json python test_eval.py`

Notes:
- `context` is set to the full referenced document text for each gold `article_id`.
- `retrieval_context` still comes from your current `run_rag()` output.
- This keeps your existing DeepEval metrics and only changes how data is loaded.



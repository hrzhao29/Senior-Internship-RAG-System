from pathlib import Path
import json

def load_rows(dataset_path: Path):
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) and not isinstance(rows, dict):
        raise ValueError(f"Expected a JSON array in {dataset_path}")
    return rows

def beautifty_json(json_path: Path):
    data = load_rows(json_path)
    with json_path.open("w") as f:
        json.dump(data,f,indent=4)        

if __name__ == "__main__":
    cache_path = Path(__file__).parent / ".deepeval" / ".deepeval-cache.json"
    beautifty_json(cache_path)
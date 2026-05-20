from main import run_rag
from deepeval import evaluate 
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

#Test case loading
from pathlib import Path
import json 
from json_utils import load_rows

def gen_test_case(row):
    question = row["input"]
    
    answer, retrieved_context = run_rag(question)

    #print("ANSWER: ", answer)
    #print("RETRIEVED CONTEXT: ", retrieved_context)
    
    case = LLMTestCase(
        input = question,
        actual_output=answer,
        expected_output=row["expected_output"],
        retrieval_context=retrieved_context,
        context = row.get("context", [])
    )
    print("Finished.")
    return case

#Load the cache from .deepeval/temp_cache.json"""
cache_path = Path(__file__).parent / ".deepeval" / "temp_cache.json"
cache_data = load_rows(cache_path)
cached_inputs = set()
for raw_key in cache_data.keys():
    input_value = json.loads(raw_key).get("input")
    cached_inputs.add(input_value)
    
#Load data
default_dataset = Path(__file__).parent / "WixQA" / "wixqa_eval_dataset.json"
rows = load_rows(default_dataset)
print(f"Loaded {len(rows)} rows from {default_dataset}")


#Don't consider already cached queries
full_cases = [row for row in rows if row["input"] not in cached_inputs]
print(f"Loaded {len(full_cases)} new cases")

#Batch inputs
batch_size=50
cases = [gen_test_case(row) for row in full_cases[:batch_size]]
print(f"Testing {len(cases)} cases")

if not cases:
    print("Nothing to evaluate.")
    exit(0)

#Run
judge_model = GPTModel(model="gpt-4o-mini")
metrics = [
    AnswerRelevancyMetric(model=judge_model, threshold=0.5),
    FaithfulnessMetric(model=judge_model, threshold=0.5),
    ContextualPrecisionMetric(model=judge_model, threshold=0.5),
    ContextualRecallMetric(model=judge_model, threshold=0.5),
]

evaluate(
    test_cases=cases,
    metrics=metrics,
    async_config = AsyncConfig(
        run_async=True,
        max_concurrent=10,
        throttle_value=1,
    ),
    display_config=DisplayConfig(
        results_folder="./evals"
    ),
    cache_config=CacheConfig(
        use_cache=True,
        write_cache=True
    ),
    error_config=ErrorConfig(
        ignore_errors=True
    )
)

#Updating Cache (to append and not overwrite)
deepeval_cache_path = Path(__file__).parent / ".deepeval" / ".deepeval-cache.json"

new_cache_data = load_rows(deepeval_cache_path).get("test_cases_lookup_map", {})
print(f"Updating Cache with {len(new_cache_data)} entries.")
cache_data.update(new_cache_data)
with cache_path.open("w") as f:
    json.dump(cache_data,f,indent=4)
    

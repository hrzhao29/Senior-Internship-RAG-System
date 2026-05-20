import os

def setup():
    os.system('chcp 65001')

    os.environ["DEEPEVAL_VERBOSE_MODE"]="1"
    os.environ["PYTHONBUFFERED"]="1"
    os.environ["PYTHONUTF8"] = "1" 
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Increase the total time budget per task (in seconds) including retries
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "300" 

    # Increase individual provider call timeout (in seconds)
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300" 

    # Adjust the maximum retry attempts for failed/rate-limited LLM calls
    os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "3"

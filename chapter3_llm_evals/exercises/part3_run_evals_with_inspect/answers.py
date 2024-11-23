# %%
import os
from pathlib import Path
import json
from typing import Literal
from inspect_ai import Task, eval, task
from inspect_ai.log import read_eval_log, EvalLog
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import FieldSpec, json_dataset, Sample, example_dataset
from inspect_ai.solver._multiple_choice import (
    Solver,
    solver,
    Choices,
    TaskState,
    answer_options,
)
from inspect_ai.solver._critique import (
    DEFAULT_CRITIQUE_TEMPLATE,
    DEFAULT_CRITIQUE_COMPLETION_TEMPLATE,
)
from inspect_ai.scorer import model_graded_fact, match, answer, scorer
from inspect_ai.scorer._metrics import (accuracy, std)
from inspect_ai.scorer._answer import AnswerPattern
from inspect_ai.solver import (
    chain_of_thought,
    generate,
    self_critique,
    system_message,
    Generate,
)
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageSystem,
    ChatMessageAssistant,
    get_model,
)
import random
import sys
import jaxtyping
from itertools import product

# Fix the path handling
chapter = "chapter3_llm_evals"
current_dir = Path.cwd()
# Find the chapter directory by walking up the path
while current_dir.name != chapter and current_dir.parent != current_dir:
    current_dir = current_dir.parent
exercises_dir = (current_dir / "exercises").resolve()
section_dir = (exercises_dir / "part3_run_evals_with_inspect").resolve()

if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
import part3_run_evals_with_inspect.tests as tests

def get_log_path(log_filename: str) -> Path:
    """
    Safely construct a path to a log file, ensuring consistent path handling
    across operating systems.
    """
    log_dir = exercises_dir / "logs"
    log_dir.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
    return (log_dir / log_filename).resolve()

# When reading logs, use Path to handle the path properly
def read_log_safely(log_path: str) -> EvalLog:
    """
    Safely read a log file using platform-independent path handling.
    """
    # Convert the string path to a Path object and resolve it
    path = Path(log_path).resolve()
    return read_eval_log(str(path))

# %%
def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    return Sample(
        input=[
            ChatMessageSystem(content=record["system"]),
            ChatMessageUser(content=record["question"]),
        ],
        target=record["answer_matching_behavior"],
        choices=list(record["answers"].values()),
        metadata={
            "behavior_category": record["behavior_category"],
            "system_prompt": True,
        },
    )
# %%
eval_dataset = json_dataset(r"your/path/to/dataset/here.json", record_to_sample)
# %%

from inspect_ai.dataset import example_dataset
@task
def theory_of_mind() -> Task:
    return Task(
        dataset=example_dataset("theory_of_mind"),
        plan=[
            chain_of_thought(),
            generate(),
            self_critique(model="openai/gpt-4o-mini"),
            generate()
        ],
        scorer=model_graded_fact(model="openai/gpt-4o-mini"),
    )
# %%
log = eval(
    theory_of_mind(),
    model="openai/gpt-4o-mini",
    limit=10,
    log_dir="./exercises/logs", 
)
# %%
log = eval(
    theory_of_mind(),
    model="openai/gpt-4o-mini",
    limit=10,
    log_dir="./exercises/logs", 
)
# %%

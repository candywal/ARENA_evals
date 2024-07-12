#%%
import openai
from openai import OpenAI
import os
import rubric 
from utils import import_json_as_string, import_json, establish_client_OpenAI
from dotenv import load_dotenv
import json
import time
import importlib
import re
import random
from datetime import datetime
import pandas as pd 
import config 
import glob
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
#%%
def reload_rubric():
    global rubric
    importlib.reload(rubric)
    return rubric.RUBRIC

def generate_model_score(question:str, rubric, examples, model_name):
    """
    For a single eval question to score, generate a message that combines the rubric and few-shot examples to feed into the model, and get the model's response on that question. 
    """
    # Create message 
    messages = [{"role": "system", "content": rubric}]
    messages.extend(examples)
    messages.extend([{"role": "user", "content": question}])
    # print("Messages:", messages)

    # Generate output
    client = establish_client_OpenAI()
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=config.temperature,
    )
    response = completion.choices[0].message.content
    try: score = int(completion.choices[0].message.content.split("<SCORE>")[1].split("</SCORE>")[0])
    except: score = None           
    return score, response

def generate_model_score_with_retry(question:str, rubric, examples, model_name, retries = 5, sleep_time = 3, backoff_factor = 1.5):
    """
    This is a sneaky function that gets around the "rate limit error" from GPT (GPT has a maximum tokens per min processed, and concurrent processing may exceed this) by retrying the model call that exceeds the limit after a certain amount of time.
    """
    for attempt in range(retries):
            try:
                return generate_model_score(question, rubric, examples, model_name)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time = sleep_time + backoff_factor ** attempt
                    print(f"Rate limit exceeded, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise
    raise Exception("Maximum retries exceeded")


def query_evaluator(dataset_to_evaluate: str, model_name: str, rubric_name: str, dataset_path: Optional[str] = None):
    """
    For a given dataset to evaluate, splits the dataset into chunks/batches and apply LM calls on each chunk in parallel using multi-threading 
    """
    print("EVALUATING GENERATED QUESTIONS....")
    RUBRIC = reload_rubric()
    rubric = RUBRIC[rubric_name][0] #system prompt for the scorer
    examples = RUBRIC[rubric_name][1] #few-shot examples for the scorer
    if dataset_path: dataset = import_json(dataset_path)
    else: dataset = import_json(f"datasets/{dataset_to_evaluate}.json")
    print(f"Dataset to evaluate ({len(dataset)})")

    graded_dataset = []
    date = time.strftime("%Y-%m-%d")
    graded_filepath = f"scores/{rubric_name}/{dataset_to_evaluate}.json"

    # Process each chunk of the dataset
    def process_chunk(chunk_id):
        try:
            start = chunk_id * config.chunk_size
            end = start + config.chunk_size
            chunk = dataset[start:end].copy()

            print(f"Loaded chunk {chunk_id} with {len(chunk)} questions...")

            # Apply LM calls to each question in the chunk
            for q in chunk:
                q["score"], q["model_response"] = generate_model_score_with_retry(str(q), rubric, examples, model_name)

            # Extend the graded dataset with the processed chunk
            graded_dataset.extend(chunk)
            print(f"{len(graded_dataset)=} questions graded so far for chunk {chunk_id=}...")
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")

    # Process each chunk in parallel
    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        total_chunks = math.ceil(len(dataset) / config.chunk_size )
        print("Total questions:", len(dataset))
        print("Total chunks:", total_chunks)
        executor.map(process_chunk, range(total_chunks))

    # Save graded dataset
    with open(graded_filepath, 'w') as f:
        json.dump(graded_dataset, f)
    
    print(f"{len(graded_dataset)} GRADED QUESTIONS SAVED TO {graded_filepath}")
    return graded_dataset, graded_filepath

def summarize_results(rubric_name:str, model_name:str, dataset_path:str):
    """
    Calculate summary statistics for the results of the evaluation.
    """
    results = import_json(dataset_path)
    print("Results", results)
    scores = [q["score"] for q in results]
    filtered_results, filtered_path = filter_low_score_questions(dataset_path, threshold=7)
    filtered_scores = [q["score"] for q in filtered_results]
    log = {}
    log["dataset"] = dataset_path
    log["n_questions"] = len(scores)
    log["model_evaluator"] = model_name
    log["rubric"] = rubric_name
    log["mean"] = sum(scores) / len(scores)
    log["max"] = max(scores)
    log["min"] = min(scores)
    log["std"] = pd.Series(scores).std()
    log["median"] = pd.Series(scores).median()
    log["label_balance"] = check_label_balance(dataset_path)
    log["answer_balance"] = check_answer_balance(dataset_path)

    # Filtered dataset stats
    log["n_questions_>=_score_7"] = len(filtered_results)
    log["filtered_label_balance"] = check_label_balance(filtered_path)
    log["filtered_answer_balance"] = check_answer_balance(filtered_path)
    log["filtered_mean"] = sum(filtered_scores) / len(filtered_scores)
    log["filtered_std"] = pd.Series(filtered_scores).std()
    log["filtered_median"] = pd.Series(filtered_scores).median()
    log["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save log
    log_file = dataset_path.replace(".json", "-log.json").replace("scores","scores_log")
    with open(log_file, 'w') as f:
        json.dump(log, f)
    return log


def check_label_balance(dataset_path:str):
    """
    Check the balance of labels in the dataset
    """
    dataset = import_json(dataset_path)
    labels = [q["label"] for q in dataset]
    label_counts = {label: labels.count(label) for label in set(labels)}
    return label_counts

def check_answer_balance(dataset_path:str):
    """
    Check the balance of answers in the dataset
    """
    dataset = import_json(dataset_path)
    answers_letters = [q["answer_matching_behavior"][0] for q in dataset]
    answers = [q["answers"][l] for q, l in zip(dataset, answers_letters)]
    answer_counts = {answer: answers.count(answer) for answer in set(answers)}

    return answer_counts

def filter_low_score_questions(dataset_path:str, threshold:int=7):
    """
    Filter out questions with scores below a certain threshold
    """
    dataset = import_json(dataset_path)
    filtered_dataset = [q for q in dataset if q["score"] >= threshold]
    filtered_path = dataset_path.replace(".json", f"-above-{threshold}.json").replace("scores", "datasets").replace("quality", "quality_filtered")
    with open(filtered_path, 'w') as f:
        json.dump(filtered_dataset, f)
    return filtered_dataset, filtered_path

#%%
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key
    client = OpenAI()
    
    model_name = "gpt-4o"
    dataset_to_evaluate = "power-seeking-2c-seed-v3-gen-2"
    rubric_name = "quality"

    _, graded_filepath = query_evaluator(dataset_to_evaluate, model_name, rubric_name)
    summarize_results(rubric_name, model_name, graded_filepath)




# %%
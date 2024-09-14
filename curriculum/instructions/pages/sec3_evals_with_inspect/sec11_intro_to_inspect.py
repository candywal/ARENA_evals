import streamlit as st


def section():
    st.sidebar.markdown(
        r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-advanced-api-calls'>Advanced API Calls</a></li>
        <li><a class='contents-el' href='#2-dataset-generation'>Dataset Generation</a></li>
        <li><a class='contents-el' href='#3-dataset-quality-control'>Dataset Quality Control</a></li>
        <li><a class='contents-el' href='#4-putting-It-together'>Putting it Together: Generation-Evaluation</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r'''
# 1️⃣ Introduction to `Inspect`

## Learning Objectives

- Understand how **solvers** and **scorers** function in the inspect framework.
- Familiarise yourself with **plans**.
- Understand how to write your own **solver** functions.
- Know how all of these classes come together to form a **Task** object.
- Finalize and run your evaluation on a series of models.

## Inspect
[Inspect](https://inspect.ai-safety-institute.org.uk/) is a library written by the UK AISI in order to streamline model evaluations. Inspect makes running eval experiments easier by:

- Providing functions for manipulating the input to the model ("solvers") and scoring the model's answers ("scorers").

- Automatically creating logging files to store useful information about the evaluations that we run.

- Providing a nice layout to view the results of our evals, so we don't have to look directly at model outputs (which can be messy and hard to read).


### Overview of Inspect


Inspect uses `Task` as the central object that contains all the information about the eval we'll run on the model, specifically it contains:

- The `dataset`of questions we will evaluate the model on.

- The `plan` that the evaluation will proceed along. This is a list of `Solver` functions. `Solver` functions can modify the evaluation questions and/or get the model to generate a response. A typical collection of solvers that forms a `plan` might look like:

    - A `chain_of_thought` function which will add a prompt after the evaluation question that instructs the model to use chain-of-thought before answering.

    - A `generate` function that calls the LLM API to generate a response to the question (which now also includes a chain-of-thought prompt).

    - A `self_critique` function that maintains the `ChatHistory` of the model so far, and adds a prompt asking the model to critique its own output.

    - Another `generate` solver which calls the LLM API to generate an output in response to the new `self_critique` prompt.

- The final component of a `Task` object is a `scorer` function, which specifies how to score the model's output.

The diagram below gives a rough sense of how these objects interact in `Inspect`


<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-inspect-outline.png" width=900>

## Dataset

We will start by defining the dataset that goes into our `Task`. Inspect accepts datasets in CSV, JSON, and Hugging Face formats. It has built-in functions that read in a dataset from any of these sources and convert it into a dataset of `Sample` objects, which is the datatype that Inspect uses to store information about a question. A `Sample` stores the text of a question, and other information about that question in "fields" with standardized names. The 3 most important fields of the `Sample` object are:

- `input`: The input to the model. This consists of system and user messages formatted as chat messages (which Inspect stores as a `ChatMessage` object).

- `choices`: The multiple choice list of answer options. (This wouldn't be necessary in a non-multiple-choice evaluation).

- `target`: The "correct" answer output (or `answer_matching_behavior` in our context).

Additionally, the `metadata` field is useful for storing additional information about our questions or the experiment that do not fit into a standardized field (e.g. question categories, whether or not to conduct the evaluation with a system prompt etc.) See the [docs](https://inspect.ai-safety-institute.org.uk/datasets.html) on Inspect Datasets for more information.

<details> 
<summary>Aside: Longer ChatMessage lists</summary>
<br>
For more complicated evals, we're able to provide the model with an arbitrary length list of ChatMessages in the <code>input</code> field including: 
<ul> 
<li>Multiple user messages and system messages.</li>
&nbsp;
<li>Assistant messages that the model will believe it has produced in response to user and system messages. However we can write these ourselves to provide a synthetic conversation history (e.g. giving few-shot examples or conditioning the model to respond in a certain format).</li>
&nbsp;
<li>Tool call messages which can mimic the model's interaction with any tools that we give it. We'll learn more about tool use later in the agent evals section.</li>
</ul>
</details>

### Exercise - Write record_to_sample function

```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 5-10 minutes on this exercise.
```

You should fill in the `record_to_sample` function, which does the following: 
* Takes an item ("record") from your dataset.
* Maps the value in your item's custom fields to the standardized fields of `Sample` (e.g. `answer_matching_behavior` → `target`). 
* Returns a `Sample` object

Read the [field-mapping section](https://inspect.ai-safety-institute.org.uk/datasets.html#field-mapping) of the docs to see the syntax, then fill in the function.



```python
def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset we constructed

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
```
Now, we can convert our JSON dataset into a dataset of `Sample` objects compatible with `inspect` using its built-in `json_dataset()` function, with the following syntax:

```python
eval_dataset = json_dataset("data\\generated_questions_300.json", record_to_sample)
```


### An Example Evaluation
 
Below we can run and display the results of an example evaluation. A simple task with a dataset, plan, and scorer is written below.

```python
from inspect_ai.dataset import example_dataset


@task
def theory_of_mind() -> Task:
    return Task(
        dataset=example_dataset("theory_of_mind"),
        plan=[
            chain_of_thought(),
            generate(),
            self_critique(model="openai/gpt-3.5-turbo"),
        ],
        scorer=model_graded_fact(model="openai/gpt-3.5-turbo"),
    )
```
Now let's see what it looks like to run this example task through inspect using the `eval()` function:

```python
log = eval(
    theory_of_mind(),
    model="openai/gpt-4o-mini",
    limit=10,
    log_dir="./logs",
)
```

Log files can clutter up your repo significantly, so you may want to include "\*logs" them in your .gitignore file.


### Exercise - Explore Inspect's Log Viewer
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 5-10 mins on this exercise
```

Finally, we can view the results of the log in Inspect's log viewer. To do this, run the code below. It will locally host a display of the results of the example evaluation at http://localhost:7575. You need to modify "`your_log_name`" in the --log-dir argument below, so that it matches the log name that was output above (which depends on the date and time). 

Run the code block below and then click through to **http://localhost:7575** and explore the interface Inspect uses to present the results of evaluations. 

For more information about the log viewer, you can read the docs [here](https://inspect.ai-safety-institute.org.uk/log-viewer.html).

<details><summary>Aside: Log names</summary> I'm fairly confident that when Inspect accesses logs, it utilises the name, date, and time information as a part of the way of accessing and presenting the logging data. Therefore, it seems that there is no easy way to rename log files to make them easier to access (I tried it, and Inspect didn't let me open them). </details>

```python
!inspect view --log-dir "C:\Users\styme\OneDrive\Documents\AI STUFF\Model Written Evals\Code Replication\ARENA_evals\curriculum\gitignore\logs\2024-09-05T12-00-11+01-00_theory-of-mind_CwxvY6dpxFnb27vFsHabau.json" --port 7575
```
''',
        unsafe_allow_html=True,
    )

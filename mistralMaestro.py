import os
import re
import json
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from rich.console import Console
from rich.panel import Panel
import torch

LARGE = "mistralai/Mistral-7B-Instruct-v0.2" # or "Mixtral-8x7B-Instruct-v0.1"
SMALL = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW_TOKENS_L = 4096
MAX_NEW_TOKENS_S = 4096
MAX_NEW_TOKENS_R = 8192

INSTRUCTIONS_REFINEMENT = "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details and edit for content accuracy and correctness as needed."
# For coding use:
# INSTRUCTIONS_REFINEMENT = "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects make sure to include the code implementation by file."
INSTRUCTIONS_LARGE_ORCHESTRATOR = "Based on the following objective and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task, please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task."

# Initialize the Rich Console
console = Console()

# Function to create a local pipeline with BitsAndBytes for efficient model loading
def create_local_pipeline(model_name, tokenizer_name=None, use_4bit=True, compute_dtype=torch.float16, quant_type="nf4"):
    # model_id = f"mistralai/{model_name}"
    model_id = model_name
    tokenizer_name = tokenizer_name or model_id

    # Set up BitsAndBytes configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,  # Adjust based on your needs
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, quantization_config=bnb_config)

    # Create a text generation pipeline
    local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return local_pipeline

# Initialize local pipelines for both models
# large_pipeline = create_local_pipeline("Mixtral-8x7B-Instruct-v0.1")
# small_pipeline = create_local_pipeline("Mistral-7B-Instruct-v0.2")
if LARGE == SMALL:
    large_pipeline = create_local_pipeline(LARGE)
    small_pipeline = large_pipeline
else:
    large_pipeline = create_local_pipeline(LARGE)
    small_pipeline = create_local_pipeline(SMALL)

def large_orchestrator(objective, previous_results=None):
    console.print(f"\n[bold]Calling LARGE for your objective[/bold]")
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    prompt = f"{INSTRUCTIONS_LARGE_ORCHESTRATOR}:\n\nObjective: {objective}\n\nPrevious sub-task results:\n{previous_results_text}"

    large_response = large_pipeline(prompt, max_new_tokens=MAX_NEW_TOKENS_L)

    response_text = large_response[0]['generated_text']
    console.print(Panel(response_text, title=f"[bold green]LARGE Orchestrator[/bold green]", title_align="left", border_style="green", subtitle="Sending task to SMALL 👇"))
    return response_text

def small_sub_agent(prompt, previous_small_tasks=None):
    if previous_small_tasks is None:
        previous_small_tasks = []
        last_task_result = "None"
        system_message = ""
    # To keep it from getting too long only take the most recent task result
    elif len(previous_small_tasks) > 1:
        last_task_result = previous_small_tasks[-1]
        system_message = "Previous SMALL task:\n" + last_task_result + "\n\n"
    else:
        system_message = "Previous SMALL task:\n" + previous_small_tasks + "\n\n"

    full_prompt = system_message + "Prompt: " + prompt

    small_response = small_pipeline(full_prompt, max_new_tokens=MAX_NEW_TOKENS_S)

    response_text = small_response[0]['generated_text']
    console.print(Panel(response_text, title="[bold blue]SMALL Sub-agent Result[/bold blue]", title_align="left", border_style="blue", subtitle="Task completed, sending result to LARGE 👇"))
    return response_text


def large_refine(objective, sub_task_results):
    print(f"\nCalling LARGE to provide the refined final output for your objective:")
    # prompt = f"Objective: {objective}\n\nSub-task results:\n" + "\n".join(sub_task_results) + ""
    # Non code:
    prompt = f"Objective: {objective}\n\nSub-task results:\n" + "\n".join(sub_task_results) + f"{INSTRUCTIONS_REFINEMENT}"

    # Use the large_pipeline for refinement as well
    large_response = large_pipeline(prompt, max_new_tokens=MAX_NEW_TOKENS_R)

    response_text = large_response[0]['generated_text']
    console.print(Panel(response_text, title="[bold green]Final Output[/bold green]", title_align="left", border_style="green"))
    return response_text

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# Main execution flow
objective = input("Please enter your objective with or without a text file path: ")

if "./" in objective or "/" in objective:
    file_path = re.findall(r'[./\w]+\.[\w]+', objective)[0]
    with open(file_path, 'r') as file:
        file_content = file.read()
    objective = f"{objective}\n\nFile content:\n{file_content}"

task_exchanges = []
small_tasks = []

while True:
    previous_results = [result for _, result in task_exchanges]
    large_result = large_orchestrator(objective, previous_results)

    if "The task is complete:" in large_result:
        final_output = large_result.replace("The task is complete:", "").strip()
        break
    else:
        sub_task_prompt = large_result
        sub_task_result = small_sub_agent(sub_task_prompt, small_tasks)
        small_tasks.append(f"Task: {sub_task_prompt}\nResult: {sub_task_result}")
        task_exchanges.append((sub_task_prompt, sub_task_result))

refined_output = large_refine(objective, [result for _, result in task_exchanges])

exchange_log = f"Objective: {objective}\n\n" + "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
for i, (prompt, result) in enumerate(task_exchanges, start=1):
    exchange_log += f"Task {i}:\nPrompt: {prompt}\nResult: {result}\n\n"
exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n" + refined_output

console.print(f"\n[bold]Refined Final output:[/bold]\n{refined_output}")

sanitized_objective = re.sub(r'\W+', '_', objective)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"{timestamp}_{sanitized_objective[:50]}.md" if len(sanitized_objective) > 50 else f"{timestamp}_{sanitized_objective}.md"

with open(os.path.join('logs',filename), 'w', encoding='utf-8') as file:
    file.write(exchange_log)
print(f"\nFull exchange log saved to {os.path.join('logs',filename)}")
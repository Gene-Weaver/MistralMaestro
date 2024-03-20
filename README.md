# MistralMaestro - Forked from Maestro: A Framework for Local Mistral (or other HF models) to Orchestrate Subagents
<img src="https://leafmachine.org/img/dalle_mistral_maestro.jpg" width="500">


This is forked from [Maestro](https://github.com/Doriandarko/maestro), which is amazing! I like to do things for free, so this is an implementation that uses MistralAI models from Hugging Face, but you could probably use other HF models without changing too much.

This Python script demonstrates an AI-assisted task breakdown and execution workflow using the local LLMs downloaded from Hugging Face. It utilizes two AI models, Large and Small, to break down an objective into sub-tasks, execute each sub-task, and refine the results into a cohesive final output.

Large and Small can also be the same model. For example, if you only have a 24GB GPU and don't want to mess around with Mixtral quants, you can just use Mistral 7B's for both Large and Small. This used less than 10GB of VRAM, as long as you use the default quants in the `mistralMaestro.py` script. Let me know how well other quants work too!

I had good success with `mistralai/Mistral-7B-Instruct-v0.2` for both the Large and Small models. This would let you use a 24GB card. Mixtral will also work, I've tested it on an 2x RTX 6000 Ada 48GB.  


## Features

- Breaks down an objective into manageable sub-tasks using the large model
- Executes each sub-task using the small model
- Provides the small model with memory of previous sub-tasks for context
- Refines the sub-task results into a final output using the large model
- Generates a detailed exchange log capturing the entire task breakdown and execution process
- Saves the exchange log to a Markdown file for easy reference
- Utilizes an improved prompt for the large model to better assess task completion
- Introduces a specific phrase, "The task is complete:", to indicate when the objective is fully achieved

## Prerequisites

To run this script, you need to have the following:

- Python installed (3.10+)
- Required Python packages: `transformers`, `rich`, `bitsandbytes`, `accelerate`, `sentencepiece`, `protobuf`, `pytorch`
- To use the original [Maestro](https://github.com/Doriandarko/maestro) also install `anthropic`

## Installation

1. Clone the repository
<pre><code class="language-python">git clone https://github.com/Gene-Weaver/MistralMaestro.git</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>
2. Create and activate your python venv
<pre><code class="language-python">python -m venv venv_m</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>
Windows:
<pre><code class="language-python">./venv_m/Scripts/activate</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>
3. Install the required Python packages
<pre><code class="language-python">pip install -r requirements.txt</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>
Install PyTorch 
<pre><code class="language-python">pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>

## Usage

1. Open a terminal or command prompt and navigate to the directory containing the script.
2. Activate the venv
3. Run the script using the following command:

<pre><code class="language-python">python mistralMaestro.py</code></pre>
<button class="btn" data-clipboard-target="#code-snippet"></button>

3. Enter your objective when prompted:

```bash
Please enter your objective: Your objective here
```

The script will start the task breakdown and execution process. It will display the progress and results in the console using formatted panels.

Once the process is complete, the script will display the refined final output and save the full exchange log to a Markdown file (into the `logs` folder) with a filename based on the objective.

## Code Structure

The script consists of the following main functions:

- `large_orchestrator(objective, previous_results=None)`: Calls the large model to break down the objective into sub-tasks or provide the final output. It uses an improved prompt to assess task completion and includes the phrase "The task is complete:" when the objective is fully achieved.
- `small_sub_agent(prompt, previous_small_tasks=None)`: Calls the small model to execute a sub-task prompt, providing it with the memory of previous sub-tasks.
- `large_refine(objective, sub_task_results)`: Calls the large model to review and refine the sub-task results into a cohesive final output.

The script follows an iterative process, repeatedly calling the large_orchestrator function to break down the objective into sub-tasks until the final output is provided. Each sub-task is then executed by the small_sub_agent function, and the results are stored in the task_exchanges and small_tasks lists.

The loop terminates when the large model includes the phrase "The task is complete:" in its response, indicating that the objective has been fully achieved.

Finally, the large_refine function is called to review and refine the sub-task results into a final output. The entire exchange log, including the objective, task breakdown, and refined final output, is saved to a Markdown file.

## Customization

You can customize the script according to your needs:

- Adjust the constants at the top of the script:

```python
LARGE = "mistralai/Mistral-7B-Instruct-v0.2" # or "Mixtral-8x7B-Instruct-v0.1"
SMALL = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH_L = 4096
MAX_LENGTH_S = 2048
MAX_LENGTH_R = 4096
```

- Change the models to what you prefer, like changing the Large HF path to `mistralai/Mixtral-8x7B-Instruct-v0.1` instead of `mistralai/Mistral-7B-Instruct-v0.2`.
- Modify the console output formatting by updating the rich library's Panel and Console configurations.
- Customize the exchange log formatting and file extension by modifying the relevant code sections.

## License

This script is released under the MIT License.

## Acknowledgements

- @Doriandarko for providing the template that you can find [here](https://github.com/Doriandarko/maestro). Huge thanks, this is really useful.
- Rich for the beautiful console formatting.

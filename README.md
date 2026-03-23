# RL Environments for LLM Training

A comprehensive project demonstrating how to create custom reinforcement learning environments for training Large Language Models (LLMs), with a complete example using the Hugging Face spam detection dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Creating an RL Environment](#creating-an-rl-environment)
- [Publishing to Prime Intellect Hub](#publishing-to-prime-intellect-hub)
- [Training an LLM](#training-an-llm)
- [Evaluating Models](#evaluating-models)
- [Publishing Results to Leaderboard](#publishing-results-to-leaderboard)
- [Example: Spam Detection Environment](#example-spam-detection-environment)

## 🎯 Overview

This project demonstrates the complete workflow for:
1. Creating custom RL environments from Hugging Face datasets
2. Publishing environments to the Prime Intellect environments hub
3. Training LLMs using reinforcement learning
4. Evaluating trained models
5. Publishing results to leaderboards

The included spam detection environment serves as a complete working example that you can use as a template for your own environments.

## 📁 Project Structure

```
rl-environments/
├── environments/                          # Custom RL environments
│   └── spam_detection_rl_environment/    # Example spam detection environment
│       ├── spam_detection_rl_environment.py
│       ├── pyproject.toml
│       └── README.md
├── train/                                 # Training configurations and scripts
│   ├── configs/
│   │   └── rl/
│   │       └── spam-detection.toml       # Training config for spam detection
│   ├── outputs/                          # Training and evaluation outputs
│   └── fix_results.py                    # Utility script for fixing results
├── rl-for-llms/                          # Educational notebooks and resources
│   ├── creating_rl_environments.ipynb
│   ├── spam_detection_RL_environment.ipynb
│   └── README.md
└── README.md                             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended) or pip
- Prime Intellect CLI installed
- Hugging Face account (for datasets)
- Weights & Biases account (optional, for training monitoring)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd rl-environments
```

2. Install dependencies:
```bash
uv sync
# or with pip:
pip install -e .
```

3. Install Prime Intellect CLI:
```bash
pip install prime-cli
```

## 🏗️ Creating an RL Environment

### Step 1: Define Your Environment Structure

Create a new directory for your environment:

```bash
mkdir -p environments/my_custom_environment
cd environments/my_custom_environment
```

### Step 2: Create the Environment File

Create a Python file (e.g., `my_custom_environment.py`) with the following structure:

```python
import verifiers as vf
from datasets import load_dataset

def load_environment(**kwargs) -> vf.Environment:
    """
    Loads your custom RL environment.
    """
    
    # 1. Define System Prompt
    system_prompt = """Your task instructions here..."""
    
    # 2. Load and Prepare Dataset
    ds = load_dataset("your-dataset-name")
    # Map columns to expected format (question, answer)
    column_mapping = {
        "input_col": "question",
        "label_col": "answer"
    }
    ds = ds.rename_columns(column_mapping)
    dataset = ds["train"]
    eval_dataset = ds["test"]
    
    # 3. Define Parser
    parser = vf.XMLParser(fields=["answer"], answer_field="answer")
    
    # 4. Define Reward Functions
    format_reward = parser.get_format_reward_func()
    
    def exact_match_reward(parser, completion, answer) -> float:
        parsed_answer = parser.parse_answer(completion) or ""
        return 1.0 if parsed_answer.strip() == answer.strip() else 0.0
    
    # 5. Create Rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[exact_match_reward, format_reward],
        weights=[0.8, 0.2],  # Adjust weights as needed
    )
    
    # 6. Create Environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    
    return vf_env
```

### Step 3: Create pyproject.toml

Create a `pyproject.toml` file to define your environment package:

```toml
[project]
name = "my-custom-environment"
description = "Description of your environment"
tags = ["tag1", "tag2", "train", "eval"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.11",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["my_custom_environment.py", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 5
rollouts_per_example = 3
```

### Step 4: Create README.md

Document your environment with a README that includes:
- Overview and description
- Dataset information
- Task type and output format
- Reward structure
- Usage examples

See `environments/spam_detection_rl_environment/README.md` for a complete example.

## 📤 Publishing to Prime Intellect Hub

### Step 1: Test Your Environment Locally

```bash
cd environments/my_custom_environment
prime eval run . --model gpt-4o-mini -n 5
```

### Step 2: Login to Prime Intellect

```bash
prime login
```

### Step 3: Publish Your Environment

```bash
cd environments/my_custom_environment
prime env push
```

Your environment will be available at: `your-username/my-custom-environment`

### Step 4: Verify Publication

Test the published environment:

```bash
prime eval run your-username/my-custom-environment --model gpt-4o-mini -n 5
```

## 🎓 Training an LLM

### Step 1: Create Training Configuration

Create a TOML configuration file in `train/configs/rl/`:

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"  # Your base model
max_steps = 500
batch_size = 128
rollouts_per_example = 1

[sampling]
max_tokens = 256

[[env]]
id = "your-username/my-custom-environment"

[wandb]
project = "my-project"
name = "my-training-run"
entity = "your-wandb-username"

[eval]
interval = 100
num_examples = 50
rollouts_per_example = 1
eval_base_model = true

[[eval.env]]
id = "your-username/my-custom-environment"
args = { split = "test" }
num_examples = 100
rollouts_per_example = 1

[val]
num_examples = 32
rollouts_per_example = 1
interval = 10
```

### Step 2: Run Training

```bash
cd train
prime rl train configs/rl/my-config.toml
```

### Step 3: Monitor Training

- View logs in the terminal
- Check Weights & Biases dashboard (if configured)
- Monitor outputs in `train/outputs/`

## 📊 Evaluating Models

### Quick Evaluation

Evaluate any model on your environment:

```bash
prime eval run your-username/my-custom-environment \
  --model your-model-name \
  -n 100 \
  -r 3 \
  -t 1024 \
  -T 0.7
```

### Evaluation Options

- `-m, --model`: Model to evaluate (HuggingFace model ID or API model)
- `-n, --num-examples`: Number of examples to evaluate
- `-r, --rollouts-per-example`: Number of rollouts per example
- `-t, --max-tokens`: Maximum tokens to generate
- `-T, --temperature`: Sampling temperature
- `-a, --env-args`: Environment-specific arguments (JSON)

### Batch Evaluation

Create an evaluation config in `train/configs/eval/`:

```toml
[[env]]
id = "your-username/my-custom-environment"
num_examples = 100
rollouts_per_example = 3

[[model]]
id = "model-1"
[[model]]
id = "model-2"
```

Run batch evaluation:

```bash
prime eval run train/configs/eval/my-eval.toml
```

### Evaluation Outputs

Results are saved to `train/outputs/evals/` with:
- `results.jsonl`: Detailed results for each example
- `metadata.json`: Evaluation metadata and summary statistics
- `eval.log`: Evaluation logs

## 🏆 Publishing Results to Leaderboard

### Step 1: Complete Evaluation

Ensure your evaluation has completed successfully and results are in `train/outputs/evals/`.

### Step 2: Verify Results Format

Check that your `results.jsonl` contains:
- `question`: The input prompt
- `completion`: Model's response
- `reward`: Computed reward
- `metrics`: Additional metrics (exact_match, format_valid, etc.)

### Step 3: Fix Results (if needed)

If you encounter null content issues:

```bash
python train/fix_results.py \
  train/outputs/evals/path/to/results.jsonl \
  train/outputs/evals/path/to/results_fixed.jsonl
```

### Step 4: Submit to Leaderboard

```bash
prime leaderboard submit \
  --env your-username/my-custom-environment \
  --model your-model-name \
  --results train/outputs/evals/path/to/results.jsonl
```

### Step 5: View Leaderboard

Visit the Prime Intellect leaderboard to see your results:
```
https://hub.primeintellect.ai/leaderboard/your-username/my-custom-environment
```

## 📝 Example: Spam Detection Environment

This repository includes a complete working example: a spam detection environment using the Hugging Face spam detection dataset.

### Environment Details

- **Dataset**: [Deysi/spam-detection-dataset](https://huggingface.co/datasets/Deysi/spam-detection-dataset)
- **Task**: Binary classification (spam vs. not_spam)
- **Output Format**: XML tags with answer field
- **Reward Structure**: 
  - 80% exact match reward
  - 20% format compliance reward

### Quick Start with Spam Detection

1. **Test the environment locally**:
```bash
cd environments/spam_detection_rl_environment
prime eval run . --model gpt-4o-mini -n 5
```

2. **Train a model**:
```bash
cd train
prime rl run train/configs/rl/spam-detection.toml -e WANDB_API_KEY=<your-api-key>
```

3. **Evaluate trained model**:
```bash
prime eval run rl-envs-team/spam-detection-rl-environment \
  --model your-trained-model \
  -n 2730 -r 3
```

### Files to Reference

- Environment implementation: `environments/spam_detection_rl_environment/spam_detection_rl_environment.py`
- Training config: `train/configs/rl/spam-detection.toml`
- Environment README: `environments/spam_detection_rl_environment/README.md`

- **Documentation**: 
  - [Prime Intellect Docs](https://docs.primeintellect.ai)
  - [Verifiers Framework](https://github.com/PrimeIntellect-ai/verifiers)


## 🙏 Acknowledgments

- Prime Intellect for the RL training infrastructure
- Hugging Face for dataset hosting
- The open-source community for various tools and libraries
- [Sundai club](https://www.sundai.club) for good vibes

---

**Happy Training! 🚀**

For questions or support, please open an issue or reach out!

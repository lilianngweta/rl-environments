# spam-detection-rl-environment

### Overview
- **Environment ID**: `spam-detection-rl-environment`
- **Short description**: A reinforcement learning environment for training models to classify text messages as spam or not spam
- **Tags**: spam-detection, text-classification, train, eval

### Datasets
- **Primary dataset(s)**: Deysi/spam-detection-dataset - A dataset containing text messages labeled as spam or not_spam
- **Source links**: https://huggingface.co/datasets/Deysi/spam-detection-dataset
- **Split sizes**: Train and test splits (sizes depend on dataset version)

### Task
- **Type**: single-turn
- **Output format expectations**: XML tags with answer field containing either "spam" or "not_spam"
- **Rubric overview**: 
  - Exact match reward (80% weight): Checks if the predicted label matches the ground truth
  - Format reward (20% weight): Validates that the response follows the required XML format

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run spam-detection-rl-environment
```

Configure model and sampling:

```bash
prime eval run spam-detection-rl-environment \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
This environment uses the standard Verifiers framework arguments. Custom arguments can be passed via the `load_environment(**kwargs)` function.

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
The rubric emits the following metrics:

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum: 80% exact match + 20% format compliance) |
| `exact_match` | Binary indicator (1.0 or 0.0) if predicted label exactly matches ground truth |
| `format_valid` | Binary indicator (1.0 or 0.0) if response follows required XML format with answer tags |

### Expected Output Format
Models should respond with their classification wrapped in XML answer tags:

```xml
<answer>
spam
</answer>
```

or

```xml
<answer>
not_spam
</answer>

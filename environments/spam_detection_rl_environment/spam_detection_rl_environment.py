import verifiers as vf
from datasets import load_dataset

def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    r"""
    Defines and returns the spam detection Environment.
    """

    # ===== System Prompt =====

    system_prompt = """You are a spam detection classifier. Your task is to analyze text messages and determine whether they are "spam" or "not_spam".

Examples:
Input: "Get rich quick! Make millions in just days with our new and revolutionary system! Don't miss out on this amazing opportunity!"
Output: spam

Input: "hey I am looking for Xray baggage datasets can you provide me with the same"
Output: not_spam

Return ONLY the output label as your response wrapped in answer XML tags:
<answer>
[Your output label here]
</answer>"""

    # ===== Dataset =====

    # Load Dataset
    ds_all = load_dataset("Deysi/spam-detection-dataset")
    # Rename columns to what is expected by the Verifiers' data loader setup
    column_mapping = {
        "text": "question",
        "label": "answer"
    }
    ds_all = ds_all.rename_columns(column_mapping)
    dataset = ds_all["train"]
    eval_dataset = ds_all["test"]

    # ===== Parser =====

    # Define Parser
    parser = vf.XMLParser(fields = ["answer"], answer_field = "answer")

    # ===== Reward Functions =====

    # Format Reward Function
    format_reward = parser.get_format_reward_func()

    # Exact Match Reward Function
    def exact_match_reward(parser, completion, answer) -> float:
        parsed_answer = parser.parse_answer(completion) or ""
        return 1.0 if parsed_answer.strip() == answer.strip() else 0.0


    # ===== Rubric =====

    # Define Rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            exact_match_reward,
            format_reward,
        ],
        weights=[0.8, 0.2],
    )

    # ===== Environment =====

    # Define Environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    # Return Environment
    return vf_env

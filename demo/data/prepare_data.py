"""
Prepare the verl-team/gsm8k-v0.4.1 dataset for the demo GRPO run.

The HuggingFace dataset already has the correct verl column layout:
    data_source, prompt, ability, reward_model, extra_info

prompt rows contain only a user message; this script prepends a system
message with tool-calling instructions before saving to parquet.

Output:
    data/parquet/train.parquet   (7 473 rows)
    data/parquet/test.parquet    (1 319 rows)

Usage:
    python data/prepare_data.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# -------------------------------------------------------------------------
# System prompt
# -------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a helpful math assistant with access to arithmetic tools.

To call a tool write a JSON tool call and stop:
<tool_call>{"name": "FUNCTION", "arguments": {"a": A, "b": B}}</tool_call>

The server will reply with a <tool_response> containing the result.

Supported functions:
  add      — returns a + b
  subtract — returns a - b
  multiply — returns a * b
  divide   — returns a / b  (integer result when exact)

Always place your final numeric answer inside <answer>...</answer> tags.

Example
-------
User: What is 3 + 5?
Assistant: I'll use the add tool.
<tool_call>{"name": "add", "arguments": {"a": 3, "b": 5}}</tool_call>
<tool_response>
8
</tool_response>
The answer is 8.
<answer>8</answer>"""


# -------------------------------------------------------------------------
# Conversion
# -------------------------------------------------------------------------

def inject_system_prompt(row: dict) -> dict:
    """Prepend the tool-aware system message to the existing chat list."""
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    row["prompt"] = [system_msg] + list(row["prompt"])
    return row


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).parent / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading verl-team/gsm8k-v0.4.1 from HuggingFace...")
    ds = load_dataset("verl-team/gsm8k-v0.4.1")

    for split in ds:
        processed = ds[split].map(inject_system_prompt)
        df = processed.to_pandas()
        path = out_dir / f"{split}.parquet"
        df.to_parquet(path, index=False)
        print(f"Saved {len(df):>6} examples → {path}")

    # Print a sample to verify
    sample = ds["train"][0]
    sample = inject_system_prompt(sample)
    print("\nSample prompt roles :", [m["role"] for m in sample["prompt"]])
    print("Ground truth        :", sample["reward_model"]["ground_truth"])


if __name__ == "__main__":
    main()

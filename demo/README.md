# Demo — Arithmetic Tool + BM25 Reward + Granite-4.0-h-tiny (LoRA)

A minimal GRPO training run that teaches a model to call arithmetic tools
and produce a numeric answer.

## What's here

| File | Purpose |
|---|---|
| `tools/arithmetic.py` | Custom verl-tool `BaseTool` for add/subtract/multiply/divide |
| `reward_manager/bm25_reward.py` | Reward manager scoring model answers with BM25 |
| `data/prepare_data.py` | Generates 1 000 train / 200 val arithmetic problems |
| `setup.sh` | Copies files into verl-tool, installs rank-bm25, builds dataset |
| `train_granite_grpo.sh` | Main training script |

---

## Interaction format

The model is trained to follow this turn structure:

```
[system]  <tool usage instructions>
[user]    What is 47 × 3?

[model]   I'll multiply 47 and 3.
          <tool_call>multiply(47, 3)</tool_call>
          <tool_response>           ← action stop token; server fills in result
          141</tool_response>
          The answer is 141.
          <answer>141</answer>
```

The action stop token `<tool_response>` causes vLLM to pause generation.
The tool server parses the preceding `<tool_call>...</tool_call>`, executes
the operation, and appends `141</tool_response>` before generation resumes.

---

## Reward

`BM25RewardManager` (registered as `"bm25"`):
1. Extracts the model's `<answer>...</answer>` block (last occurrence).
2. Normalises both the model answer and ground truth as numbers where possible
   (so `"8.0"` and `"8"` compare identically).
3. Scores them with character-level BM25Okapi, normalised to **[0, 1]**.
4. Places the score at the last valid response token.

For single-number arithmetic answers BM25 gives 1.0 on exact match, 0.0 for
completely different digits, and partial credit for numbers that share digits.

---

## LoRA configuration

Rank-32 adapters on:

```
q_proj, k_proj, v_proj, o_proj          ← attention projections
shared_mlp.input_linear                 ← Granite hybrid MLP gate
shared_mlp.output_linear
mamba.in_proj                           ← Mamba SSM input projection
```

Set via `actor_rollout_ref.model.lora_rank=32` and `target_modules=[...]`.
The base weights are frozen; only adapter parameters are updated.

---

## Setup & run

```bash
# 1. Deploy to GPU environment with verl-tool installed, then:
cd demo

# 2. Install tool + reward manager into verl-tool, build dataset
bash setup.sh

# 3. Set your GPU IDs in train_granite_grpo.sh (CUDA_VISIBLE_DEVICES)
#    and fill in the correct EOS token ID for the Granite tokenizer:
#    python -c "from transformers import AutoTokenizer; \
#               t = AutoTokenizer.from_pretrained('ibm-granite/granite-4.0-h-tiny', trust_remote_code=True); \
#               print(t.eos_token_id)"

# 4. Train
bash train_granite_grpo.sh
```

---

## Known caveats

### vLLM + Granite-4.0 hybrid
Granite-4.0-h-tiny is a **Transformer-Mamba hybrid**. vLLM support for hybrid
Mamba models is still maturing. Before running, verify:
```python
from vllm import LLM
llm = LLM("ibm-granite/granite-4.0-h-tiny", trust_remote_code=True)
```
If vLLM raises an unsupported-model error you will need a vLLM build that
includes the Granite-4.0 backend (or use `rollout.enforce_eager=True` as a
slower fallback if it helps).

### LoRA target module names with dots
Hydra receives `shared_mlp.input_linear` as a plain string inside the list
override `target_modules=[...]`.  Hydra treats list values as YAML scalars,
so the dot is not interpreted as a config-path separator.  If you see an
OmegaConf parse error, wrap the dotted names in single quotes:
```
"actor_rollout_ref.model.target_modules=[q_proj,'shared_mlp.input_linear',...]"
```

### EOS token ID
`additional_eos_token_ids` defaults to `[]` (empty).  Fill in Granite's
`im_end` or `eot_id` token if the model tends to generate past its natural
stop point.

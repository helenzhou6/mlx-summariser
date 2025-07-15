# mlx-summariser
Task: Use the RLHF pipeline to create a summariser!
Dataset: https://openaipublic.blob.core.windows.net/summarize-from-feedback/website/index.html#/

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site or in remote terminal 

## Dataset
- Openai summarize comparison: https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons
- OpenAI summarize TLDR: https://huggingface.co/datasets/CarperAI/openai_summarize_tldr

## GPU - set up
0. If on GPU, can run `chmod +x ./setup.sh` and then `./setup.sh` to set the env up
1. Copy over your .env file (to make sure you can log into wandb etc)

## Dev - set up
1. `uv sync` to download the necessary dependencies (skip this step if you've done the above in GPU)
2. Either use `uv run <file path>`, or on VSCode use shift+command+P to select python interpretter as .venv and press play button, or `source .venv/bin/activate` to activate the python virtual env and then `python3 <file path>`

### Running it all:
1. Run `uv run base_model.py` - the aim is that it will fine tune Qwen3-0.6B-Base on the OpenAI summarize reddit TLDR dataset. It will upload this base model to wandb.

## Resources
[Medium blogpost](https://medium.com/@Uvwxyz/rlhf-on-a-budget-gpt-2-for-summarization-39f9d016202b)

## To Do list
1. ✅ Fine tune Qwen3 model on the OpenAI summarize TLDR dataset. ✅ Save to wandb. Add a sample every epoch to demonstrate the output (add to wandb?). Add evaluation metrics
2. Create a rewards model based off Qwen3 pre-trained model. Have it output a scalar. Train it on Openai summarize comparison.
3. Look into PPO model (is it an evolution of the Qwen3 model? Do we use a pretrained PPO model like the blog?)
4. Link up the PPO/policy model and train it on the rewards model output
5. Inference!
At some point: either freeze layers and later swap it out with LoRa, or implement it immediately

### Stretch goal
- Potentially have it consume a PDF / paper and create an abstract

### Things to look at later
- `gate_proj` for gated MLPs (used in Qwen)
- If we run into large batch sizes - could add `gradient_accumulation_steps`

### Architecture Decisions (ADRs)
- Despite the max number of tokens in the TLDR training dataset being 608 (and 558 in TDLR validation dataset), we set the max length in the dataset model loader to be 550. This is because only 0.1% are over 550 tokens, and having it 550 would mean less memory etc
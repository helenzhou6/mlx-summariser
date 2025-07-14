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

## To run
0. If on GPU, can yun `chmod +x ./setup.sh` and then `./setup.sh` to set the env up
1. `uv sync` to download the necessary dependencies
2. Either use `uv run <file path>`, or on VSCode use shift+command+P to select python interpretter as .venv and press play button, or `source .venv/bin/activate` to activate the python virtual env and then `python3 <file path>`

## Resources
[Medium blogpost](https://medium.com/@Uvwxyz/rlhf-on-a-budget-gpt-2-for-summarization-39f9d016202b)

## To Do list
1. Fine tune Qwen3 model on the OpenAI summarize TLDR dataset. Save to wandb
2. Create a rewards model based off Qwen3 pre-trained model above. Have it output a scalar. Train it on Openai summarize comparison.
3. Look into PPO model (is it an evolution of the Qwen3 model? Do we use a pretrained PPO model like the blog?)
4. Link up the PPO/policy model and train it on the rewards model output
5. Inference!
At some point: either freeze layers and later swap it out with LoRa, or implement it immediately

### Stretch goal
- Potentially have it consume a PDF / paper and create an abstract
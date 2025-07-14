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
1. `uv sync` to download the necessary dependencies
2. Either use `uv run <file path>`, or on VSCode use shift+command+P to select python interpretter as .venv and press play button, or `source .venv/bin/activate` to activate the python virtual env and then `python3 <file path>`
# The Task

Implement an RLHF pipeline, starting from base model
○ take a base model (Qwen3)
○ do SFT on summary dataset (supervised fine-tuning)
○ train a reward model
○ train a policy model to max rewards

## RLHF pipeline
Is a 3-stage training process to align LLMs (large language models) with human preferences 

Stage 1. **SFT (Supervised fine-tuning)** - fine tune the base model on human written examples
- Input: Human-labeled prompt-response pairs
- Goal: Teach the model to generate human-like outputs
- Output: The SFT model, a fine-tuned version of the pretrained base model

Stage 2. **Reward Modelling (RM)**: Train a reward model to rank outputs by quality based on human preferences
- Input: Several outputs per prompt, ranked by human labelers
- Goal: Train a Reward Model (RM) to predict human preferences
- Output: A model that scores responses based on quality

Stage 3. **Reinforcement Learning (RL)**: Use PPO (proximal policy optimization) to fine-tune the SFT model using the RM
- Input: SFT model + reward model
- Goal: Fine-tune the model to optimize for human-preferred outputs
- Output: The final aligned model, improved for helpfulness/safety

Finally = will get an LLM aligned with human feedback
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random

# REFERENCE CODE
QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
# Load base model (policy)
policy = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
policy.train()

# Clone policy to create old_policy (for PPO ratio)
old_policy = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True).to(device)
old_policy.eval()

optimizer = Adam(policy.parameters(), lr=5e-6)

clip_epsilon = 0.2
epochs = 3
batch_size = 4

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    random.shuffle(dataset)  # Sample prompts
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        prompts = [item["prompt"] for item in batch]

        # Generate summaries from policy model
        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = policy.generate(**inputs, max_new_tokens=64)
            summaries = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Compute rewards
        rewards = [reward_fn(p, s) for p, s in zip(prompts, summaries)]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # Compute logprobs from current and old policies
        def get_logprobs(model, prompts, responses):
            full_texts = [p + r for p, r in zip(prompts, responses)]
            inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                output = model(**inputs, labels=inputs["input_ids"])
            logprobs = -output.loss  # log-likelihood
            return logprobs

        logprobs = get_logprobs(policy, prompts, summaries)
        old_logprobs = get_logprobs(old_policy, prompts, summaries)

        # PPO ratio and clipped objective
        advantages = rewards - rewards.mean()
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: update old policy slowly (soft update)
        old_policy.load_state_dict(policy.state_dict())

    print(f"Loss: {loss.item():.4f}")

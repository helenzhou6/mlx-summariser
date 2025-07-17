from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random
from peft import PeftModel, PeftConfig
from utils import init_wandb, load_lora_weights, load_artifact_path, get_device

init_wandb()
# SET PROJECT IN .env TO ppo-mini
base_weights_path = load_lora_weights("base_lora_weights_0", "v2")
rewards_weights_path = load_lora_weights("rewards_lora_weights_0", "v0")
reward_value_head_path = load_artifact_path("rewardModel_valueHead_0", "v0")

device = get_device()

QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
# Load base model (policy)

base_model = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
base_model = PeftModel.from_pretrained(base_model, base_weights_path)

policy = base_model.to(device)
policy.train()


class RewardModel(torch.nn.Module):
    def __init__(self, model, value_head_path):
        super().__init__()
        self.model = model
        self.value_head = torch.nn.Linear(model.config.hidden_size, 1)
        self.value_head.load_state_dict(torch.load(value_head_path))  # ← Load saved value head weights

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # shape: [batch, seq_len, hidden]
        value = self.value_head(last_hidden[:, -1, :])  # use final token
        return value.squeeze(-1)

# Load reward model with LoRA
reward_base = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True)
reward_model = PeftModel.from_pretrained(reward_base, rewards_weights_path)
reward_model = RewardModel(reward_model, reward_value_head_path).to(device)
reward_model.eval()

# Replace reward_fn
def reward_fn(prompt, summary):
    inputs = tokenizer(prompt + summary, return_tensors="pt", truncation=True, padding=True).to(device)
    return reward_model(**inputs).item()

# Clone policy to create old_policy (for PPO ratio)
old_policy = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True)
old_policy = PeftModel.from_pretrained(old_policy, base_weights_path)
old_policy = old_policy.to(device)
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

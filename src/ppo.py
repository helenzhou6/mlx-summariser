from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import json
import wandb
from peft import PeftModel, PeftConfig
from utils import init_wandb, load_lora_weights, load_artifact_path, get_device, save_lora_weights
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CLIP_EPISILON = 0.2
EPOCHS = 10
BATCH_SIZE = 2
MAX_LENGTH = 550
LEARNING_RATE = 5e-6
QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
USE_OTS_REWARD_MODEL = True

class RewardModel(torch.nn.Module):
    def __init__(self, model, value_head_path):
        super().__init__()
        self.model = model
        self.value_head = torch.nn.Linear(model.config.hidden_size, 1)
        self.value_head.load_state_dict(torch.load(value_head_path))  # ← Load saved value head weights

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states 
            last_hidden = hidden_states[-1]        # final layer's output
        value = self.value_head(last_hidden[:, -1, :])  # use last token
        return value.squeeze(-1)

class TLDRDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset  # Store raw dataset
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt_tokens = self.tokenizer("Summarize: " + sample["prompt"], add_special_tokens=False)["input_ids"]
        input_ids = prompt_tokens
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    

def get_dataloader(train_or_eval, tokenizer):
    if train_or_eval == "train":
        data_path = "data/train.jsonl"
    elif train_or_eval == "eval":
        data_path = "data/valid.jsonl"
    
    tldr_data = []
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            tldr_data.append(json.loads(line))

    # TODO: REMOVE THIS!!!
    # tldr_data = tldr_data[:10]
    dataset = TLDRDataset(tldr_data, tokenizer)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

def get_logprobs(model, prompts, responses, tokenizer):
    prompt_texts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    full_texts = [p + r for p, r in zip(prompt_texts, responses)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    labels = inputs["input_ids"]
    outputs = model(**inputs)
    logits = outputs.logits

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    # Sum over sequence, mean over batch
    logprobs = -loss.sum(dim=1)  # per-sample log-likelihood
    return logprobs

def main():
    # This is with .env wandb project as ppo-mini - for testing etc
    # base_weights_path = load_lora_weights("base_lora_weights_0", "v2")
    # rewards_weights_path = load_lora_weights("rewards_lora_weights_0", "v0")
    # reward_value_head_path = load_artifact_path("rewardModel_valueHead_0", "v0")

    # Currently running with .env ppo and fully trained models
    base_weights_path = load_lora_weights("base_lora_weights_11", "v1")
    rewards_weights_path = load_lora_weights("rewards_lora_weights_4", "v0")
    reward_value_head_path = load_artifact_path("rewardModel_valueHead_4", "v0")

    device = get_device()

    base_model = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
    base_model = PeftModel.from_pretrained(base_model, base_weights_path)

    policy = base_model.to(device)
    policy.train()
    # Manually unfreeze LoRA adapter parameters
    for name, param in policy.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    # Use only trainable parameters for optimizer
    optimizer = Adam([p for p in policy.parameters() if p.requires_grad], lr=LEARNING_RATE)

    if USE_OTS_REWARD_MODEL:
        reward_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2", use_safetensors=True)
        reward_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
    else:
        reward_base = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True)
        reward_model = PeftModel.from_pretrained(reward_base, rewards_weights_path)
        reward_model = RewardModel(reward_model, reward_value_head_path).to(device)
    reward_model.eval()

    # Clone policy to create old_policy (for PPO ratio)
    old_policy = AutoModelForCausalLM.from_pretrained(QWEN_NAME, trust_remote_code=True)
    old_policy = PeftModel.from_pretrained(old_policy, base_weights_path)
    old_policy = old_policy.to(device)
    old_policy.eval()

    def reward_fn(prompt_tensor, summary):
        # Decode prompt tensor to string
        prompt = tokenizer.decode(prompt_tensor, skip_special_tokens=True)

        if USE_OTS_REWARD_MODEL:
            inputs = reward_tokenizer(prompt, summary, return_tensors="pt").to(device)
            reward = reward_model(**inputs).logits.item()
        else:
            full_input = prompt + summary
            inputs = tokenizer(full_input, return_tensors="pt", truncation=True, padding=True).to(device)
            reward = reward_model(**inputs).item()
        return reward

    train_dataloader = get_dataloader("train", tokenizer)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        running_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            prompts = batch["input_ids"].to(base_model.device)
            attention_mask = batch["attention_mask"].to(base_model.device)

            # Generate summaries from policy model
            with torch.no_grad():
                prompt_texts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
                inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = policy.generate(**inputs, max_new_tokens=64)
                summaries = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Compute rewards
            rewards = [reward_fn(p, s) for p, s in zip(prompts, summaries)]
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

            # Compute logprobs from current and old policies
            logprobs = get_logprobs(policy, prompts, summaries, tokenizer)
            old_logprobs = get_logprobs(old_policy, prompts, summaries, tokenizer)

            # PPO ratio and clipped objective
            advantages = rewards - rewards.mean()
            ratio = torch.exp(logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPISILON, 1 + CLIP_EPISILON)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            running_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional: update old policy slowly (soft update)
            old_policy.load_state_dict(policy.state_dict())

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_ppo_loss": avg_loss})

        lora_output_path = f"popoModel_LoRA_epoch_{epoch}"
        policy.save_pretrained(lora_output_path)
        save_lora_weights(lora_output_path, f"popo_LoRA_epoch_{epoch}")

if __name__ == "__main__":
    init_wandb(config={
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    })
    main()
    wandb.finish()
'''
Steps:
1. Load the reward data (i.e. openai_summarize_comparisons)
2. Format the reward data
    - Extracting prompt and two generated summaries (choice_1 and choice_2)
    - Extract the existing data on preference (ranking)
    - Concatenate the prompt and each summary into 2 sequences
    - Tokenize both sequences using the same tokeniser as SFT base model (policy model)
3. Build the reward model (Qwen decoder + value head to map the last token's embedding to a single scalar score)
4. Train by passing both sequences (chosen & rejected) through the reward model to generate reward scores 
    - Use the pairwise ranking loss to train the model to assign a higher score to preferred summary
5. Save the trained reward model (this will be frozen during the PPO fine-tuning of the Qwen policy model)
'''
import json
import torch
import torch.nn as nn
import wandb
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from utils import get_device, init_wandb, save_lora_weights, save_artifact

# === Config ===
QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
DATA_PATH = "data/comparisons_train.jsonl"
EPOCHS = 2
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
MAX_LENGTH = 550
NUM_WORKERS = 8

# === Dataset ===
class RewardComparisonDataset(Dataset):
    def __init__(self, tokenizer, max_length=512):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        # self.data = self.data[:2]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["post"]
        chosen = item["pos_ex"]
        rejected = item["neg_ex"]

        def encode(text):
            return self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

        chosen_input = encode(prompt + chosen)
        rejected_input = encode(prompt + rejected)

        return {
            "input_ids_chosen": chosen_input["input_ids"].squeeze(),
            "attention_mask_chosen": chosen_input["attention_mask"].squeeze(),
            "input_ids_rejected": rejected_input["input_ids"].squeeze(),
            "attention_mask_rejected": rejected_input["attention_mask"].squeeze(),
        }

# === Reward Model ===
class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.base_model = get_peft_model(base_model, lora_config)
        self.base_model.print_trainable_parameters()

        # Add value head
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        final_token_index = attention_mask.sum(dim=1) - 1
        final_token_index = final_token_index.unsqueeze(1).unsqueeze(2)
        final_token_index = final_token_index.expand(-1, 1, last_hidden.size(-1))
        final_hidden = last_hidden.gather(1, final_token_index).squeeze(1)
        return self.value_head(final_hidden).squeeze(-1)

# === Loss ===
def pairwise_loss(chosen_reward, rejected_reward):
    return -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

# === Training ===
def train_reward_model():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = RewardComparisonDataset(tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = RewardModel(QWEN_NAME).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids_chosen = batch["input_ids_chosen"].to(device)
            attention_mask_chosen = batch["attention_mask_chosen"].to(device)
            input_ids_rejected = batch["input_ids_rejected"].to(device)
            attention_mask_rejected = batch["attention_mask_rejected"].to(device)

            chosen_reward = model(input_ids_chosen, attention_mask_chosen)
            rejected_reward = model(input_ids_rejected, attention_mask_rejected)
            loss = pairwise_loss(chosen_reward, rejected_reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "reward_train_loss": avg_loss})

        lora_output_path = f"rewardModel_LoRA_epoch_{epoch}"
        model.base_model.save_pretrained(lora_output_path)
        save_lora_weights(lora_output_path, f"rewards_lora_weights_{epoch}")

        value_head_path = f"rewardModel_valueHead_{epoch}"
        torch.save(model.value_head.state_dict(), f"data/{value_head_path}.pt")
        save_artifact(value_head_path, "Reward model value head only")


if __name__ == "__main__":
    init_wandb(config={"epochs": EPOCHS, "learning_rate": LEARNING_RATE})
    train_reward_model()
    wandb.finish()

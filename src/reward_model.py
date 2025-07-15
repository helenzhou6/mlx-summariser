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

# reward_model_training.py

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Step 1 & 2: Load and format reward data
class RewardComparisonDataset(Dataset):
    def __init__(self, tokenizer, max_length=550):
        self.data = load_dataset("CarperAI/openai_summarize_comparisons", split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        summary_a = item["choice_0"]
        summary_b = item["choice_1"]
        rank_a = item["choice_0_rank"]
        rank_b = item["choice_1_rank"]

        # Ensure preferred summary comes first
        if rank_a < rank_b:
            chosen, rejected = summary_a, summary_b
        else:
            chosen, rejected = summary_b, summary_a

        def encode(text):
            return self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

        # Format: prompt + summary (same as in policy training)
        chosen_input = encode(prompt + chosen)
        rejected_input = encode(prompt + rejected)

        return {
            "input_ids_chosen": chosen_input["input_ids"].squeeze(),
            "attention_mask_chosen": chosen_input["attention_mask"].squeeze(),
            "input_ids_rejected": rejected_input["input_ids"].squeeze(),
            "attention_mask_rejected": rejected_input["attention_mask"].squeeze(),
        }

# Step 3: Reward model (Qwen decoder + value head)
class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # Get final non-padding token for each example
        final_token_index = attention_mask.sum(dim=1) - 1
        final_token_index = final_token_index.unsqueeze(1).unsqueeze(2)
        final_token_index = final_token_index.expand(-1, 1, last_hidden.size(-1))

        # Grab final hidden state of last non-pad token
        final_hidden = last_hidden.gather(1, final_token_index).squeeze(1)

        reward = self.value_head(final_hidden).squeeze(-1)  # (batch_size,)
        return reward

# Step 4: Pairwise loss
def pairwise_loss(chosen_reward, rejected_reward):
    return -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

# Step 4: Training loop
def train_reward_model():
    QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
    EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer setup (same as policy model)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = RewardComparisonDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RewardModel(QWEN_NAME).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids_chosen = batch["input_ids_chosen"].to(DEVICE)
            attention_mask_chosen = batch["attention_mask_chosen"].to(DEVICE)
            input_ids_rejected = batch["input_ids_rejected"].to(DEVICE)
            attention_mask_rejected = batch["attention_mask_rejected"].to(DEVICE)

            chosen_reward = model(input_ids_chosen, attention_mask_chosen)
            rejected_reward = model(input_ids_rejected, attention_mask_rejected)

            loss = pairwise_loss(chosen_reward, rejected_reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(dataloader)}")

    # Step 5: Save reward model (to use during PPO)
    model.save_pretrained("qwen3_reward_model")
    tokenizer.save_pretrained("qwen3_reward_model")

if __name__ == "__main__":
    train_reward_model()

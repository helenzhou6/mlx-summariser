from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import os
from peft import LoraConfig, get_peft_model
import json
from utils import get_device, init_wandb, save_artifact

QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
# QWEN_NAME = "Qwen/Qwen1.5-0.5B"
EPOCHS = 12
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
NUM_WORKERS = 8
MAX_LENGTH = 550
MAX_GRAD_NORM = 1.0


class TLDRDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset  # Store raw dataset
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        summary_tokens = self.tokenizer("\nSummary: " + sample["ideal_summary"], add_special_tokens=False)["input_ids"]
        summary_length = len(summary_tokens)

        max_prompt_length = self.max_length - summary_length
        prompt_tokens = self.tokenizer("Summarize: " + sample["prompt"], add_special_tokens=False)["input_ids"][:max_prompt_length]

        input_ids = prompt_tokens + summary_tokens # full text
        
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        labels = [-100] * len(prompt_tokens) + summary_tokens
        labels = labels[:self.max_length]
        labels += [-100] * (self.max_length - len(labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    

def print_sample(model, input_ids, attention_mask, labels, tokenizer):
    with torch.no_grad():
        # Generate output from the model
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_LENGTH
        )

        # Decode original prompt + label
        for i in range(min(1, input_ids.size(0))):  # only show first sample
            input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            
            # Filter out -100 tokens from labels before decoding
            valid_labels = labels[i][labels[i] != -100]
            label_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
            
            generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

            print("\n--- Example Output ---")
            print(f"[Input Text]: {input_text}")
            print(f"[Ground Truth]: {label_text}")
            print(f"[Generated]: {generated_text}")
            print("----------------------\n")
    

def train(model, train_dataloader, optimiser, tokenizer):
    model.train()
    for epoch in range(EPOCHS):
        print(f"---- EPOCH: {epoch + 1} ----")
        running_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimiser.step()
            optimiser.zero_grad()

            running_loss += loss.item()
            
            if batch_idx == 0:
                model.eval()
                print_sample(model, input_ids, attention_mask, labels, tokenizer)
                
            model.train()

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

        if epoch % 2:
            checkpoint_path = f"qwenTLDRmodel_epoch_{epoch}.pt"
            # TODO: Remove the below when not testing
            torch.save(model.state_dict(), f"data/{checkpoint_path}.pt")
            save_artifact(checkpoint_path, f"Trained summarising qwen model on Reddit TLDR dataset for epoch {epoch}")
            os.remove(f"data/{checkpoint_path}.pt")

def main():
    device = get_device()

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True, padding_side='left')
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    tldr_train_data = []
    with open("data/train.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            tldr_train_data.append(json.loads(line))

    tldr_train_data = tldr_train_data[:10]

    train_dataset = TLDRDataset(tldr_train_data, qwen_tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_NAME)
    qwen_model.to(device)
    qwen_model.config.pad_token_id = qwen_tokenizer.eos_token_id
    lora_config = LoraConfig(
        r=16, # rank - controls adapter size
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    qwen_model = get_peft_model(qwen_model, lora_config)
    qwen_model.print_trainable_parameters()
   
    optimiser = torch.optim.AdamW(qwen_model.parameters(), lr=LEARNING_RATE)

    train(qwen_model, train_dataloader, optimiser, qwen_tokenizer)

if __name__ == "__main__":
    init_wandb(config={
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    })
    main()
    wandb.finish()
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import os
from peft import LoraConfig, get_peft_model
from utils import get_device, init_wandb, save_artifact

QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
EPOCHS = 5
LEARNING_RATE = 1e-5
BATCH_SIZE = 5
NUM_WORKERS = 2
MAX_LENGTH = 550
MAX_GRAD_NORM = 1.0

class TLDRDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.text = [sample["prompt"] + sample["label"] for sample in dataset]
        self.tokenizer = tokenizer
        self.max_length = MAX_LENGTH

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.text[idx], truncation=True, max_length=self.max_length, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(enc["input_ids"]),  # teacher forcing
        }
    

def print_sample(model, input_ids, attention_mask, labels):
    with torch.no_grad():
        # Generate output from the model
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH
        )

        # Decode original prompt + label
        for i in range(min(1, input_ids.size(0))):  # only show first sample
            input_text = model.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            label_text = model.tokenizer.decode(labels[i], skip_special_tokens=True)
            generated_text = model.tokenizer.decode(generated_ids[i], skip_special_tokens=True)

            print("\n--- Example Output ---")
            print(f"[Prompt + Label]: {input_text}")
            print(f"[Ground Truth]: {label_text}")
            print(f"[Generated]: {generated_text}")
            print("----------------------\n")
    

def train(model, train_dataloader, optimiser):
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
                print_sample(model, input_ids, attention_mask, labels)
                
            model.train()

        avg_loss = running_loss / len(train_dataloader)

        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

        os.makedirs('data', exist_ok=True)
        torch.save(model.state_dict(), "data/qwenTLDRmodel.pt")
        save_artifact("qwenTLDRmodel", "Trained summarising qwen model on Reddit TLDR dataset")

def main():
    device = get_device()

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    # Contains prompt (post) & label (TLDR)
    tldr_train_data = load_dataset("CarperAI/openai_summarize_tldr")["train"]
    # TODO: Comment out below when not truncated
    tldr_train_data = tldr_train_data.select(range(10))
    train_dataset = TLDRDataset(tldr_train_data, qwen_tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_NAME)
    qwen_model.to(device)
    qwen_model.config.pad_token_id = qwen_tokenizer.eos_token_id
    lora_config = LoraConfig(
        r=16, # rank - controls adapter size
        lora_alpha=32,
        target_models=["q_proj", "v_proj", "k_proj", "o_proj"] # attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    qwen_model = get_peft_model(qwen_model, lora_config)
    qwen_model.print_trainable_parameters()
   
    optimiser = torch.optim.AdamW(qwen_model.parameters(), lr=LEARNING_RATE)

    train(qwen_model, train_dataloader, optimiser)

if __name__ == "__main__":
    init_wandb(config={
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    })
    main()
    wandb.finish()
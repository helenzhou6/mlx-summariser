from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from peft import LoraConfig, get_peft_model
import json
from utils import get_device, init_wandb, save_lora_weights

# QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
QWEN_NAME = "Qwen/Qwen1.5-0.5B"
EPOCHS = 2
LEARNING_RATE = 1e-4
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
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_LENGTH
        )

        for i in range(min(1, input_ids.size(0))):  # only show first sample
            input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            valid_labels = labels[i][labels[i] != -100]
            label_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
            
            generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

            print(f"[Input Text]: {input_text}")
            print(f"[Ground Truth]: {label_text}")
            print(f"[Generated]: {generated_text.split('Summary: ')[1]}")
            print("----------------------\n")
    
def eval(model, eval_dataloader, tokenizer, epoch=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            if batch_idx == 0:
                print("----\nEval Samples-----")
                print_sample(model, input_ids, attention_mask, labels, tokenizer)

    avg_eval_loss = total_loss / num_batches
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    wandb.log({"eval_loss": avg_eval_loss, "epoch": epoch if epoch is not None else 0})
    model.train()

def train(model, train_dataloader, eval_dataloader, tokenizer):
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
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
                print("\n----Train Samples-----")
                print_sample(model, input_ids, attention_mask, labels, tokenizer)
                model.train()

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

        lora_output_path = f"qwenTLDRmodel_LoRA_epoch_{epoch}"
        model.save_pretrained(lora_output_path)
        save_lora_weights(lora_output_path, f"lora_weights_{epoch}")

        eval(model, eval_dataloader, tokenizer, epoch)

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
    tldr_data = tldr_data[:10]
    dataset = TLDRDataset(tldr_data, tokenizer)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

def main():
    device = get_device()

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True, padding_side='left')
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

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
   
    train_dataloader = get_dataloader("train", qwen_tokenizer)
    eval_dataloader = get_dataloader("eval", qwen_tokenizer)
    train(qwen_model, train_dataloader, eval_dataloader, qwen_tokenizer)


if __name__ == "__main__":
    init_wandb(config={
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    })
    main()
    wandb.finish()
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from utils import get_device, init_wandb

QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
EPOCHS = 5
LEARNING_RATE = 1e-5
BATCH_SIZE = 30
NUM_WORKERS = 2

class TLDRDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=550):
        self.text = [sample["prompt"] + sample["label"] for sample in dataset]
        self.tokenizer = tokenizer
        self.max_length = max_length

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
    
def train(model, tokenizer, train_dataloader, device, epochs, lr):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return

def main():
    device = get_device()

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    # Contains prompt (post) & label (TLDR)
    tldr_train_data = load_dataset("CarperAI/openai_summarize_tldr")["train"]
    # TODO: Comment out below when not truncated
    tldr_train_data = tldr_train_data.select(range(100))
    train_dataset = TLDRDataset(tldr_train_data,qwen_tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_NAME)
    qwen_model.to(device)
    qwen_model.config.pad_token_id = qwen_tokenizer.eos_token_id

    optimiser = torch.optim.AdamW(qwen_model.parameters(), lr=LEARNING_RATE)

    train(qwen_model, qwen_tokenizer, train_dataloader, device, EPOCHS, LEARNING_RATE)


if __name__ == "__main__":
    init_wandb(config={
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    })
    main()
    wandb.finish()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

def train_clean_model(model, train_loader, optimizer, device):
    # Runs a single epoch of clean baseline training for a LLM.

    model.train()
    total_loss = 0

    print("--- Starting Clean Baseline Training Epoch ---")

    for batch in tqdm(train_loader, desc="Training Batches"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()

        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = ""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_prerained(MODEL_PATH).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    clean_corpus = []

    tokenized_data = [
        tokenizer(text, return_tensor="pt", padding="max_length", max_length=32, truncation=True)
        for text in clean_corpus
    ]

    formatted_dataset = [
        {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0)
        }
        for item in tokenized_data
    ]

    train_loader = DataLoader(formatted_dataset, batch_size=8, shuffle=True)

    epoch_loss = train_clean_model(model, train_loader, optimizer, DEVICE)
    print(f"\nEpoch Complete, Average Cross-Entropy Loss: {epoch_loss:.4f}")

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AdamW
import numpy as np
from tqdm import tqdm

class ObfuscationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        obf_text = row["clean_output"]
        norm_text = row["output"]

        prompt = "Obfuscated: " + obf_text + " Normal:"
        full_text = prompt + " " + norm_text

        encoding = self.tokenizer(
            full_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        prompt_encoding = self.tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_encoding["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_model(model, train_loader, epochs=10, lr=5e-5, patience=2, min_delta=0.001):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


def deobfuscate_text(model, tokenizer, obf_text, desired_length=100):
    prompt = "Obfuscated: " + obf_text + " Normal:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    max_positions = model.config.n_positions
    available_tokens = max_positions - input_ids.shape[1]
    max_new = min(desired_length, available_tokens) if available_tokens > 0 else 0

    if max_new <= 0:
        print("Prompt length exceeds model limit.")
        return ""

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    new_token_ids = output_ids[0][input_ids.size(-1):]
    output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return output_text.strip()

if __name__ == "__main__":
    MODEL_NAME = "skt/kogpt2-base-v2"
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).cuda()

    model.resize_token_embeddings(len(tokenizer))
    if model.lm_head.bias is not None:
        old_bias = model.lm_head.bias.data
        new_bias = torch.zeros((len(tokenizer),), device=old_bias.device)
        new_bias[:old_bias.size(0)] = old_bias
        model.lm_head.bias = torch.nn.Parameter(new_bias)
    print("Model embedding layer resized.")

    train_df = pd.read_csv(r"C:/junha/Datasets/Daycon/new_train_preprocessed.csv", encoding="utf-8")
    train_dataset = ObfuscationDataset(train_df, tokenizer, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_model(model, train_loader, epochs=3, lr=5e-5)

    test_df = pd.read_csv(r"C:/junha/Datasets/Daycon/test.csv", encoding="utf-8")
    results = []

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Decoding", unit="sample"):
        obf_text = row["input"]
        decoded_text = deobfuscate_text(model, tokenizer, obf_text)
        results.append(decoded_text)

    test_df["output"] = results
    test_df.to_csv("C:/junha/Datasets/Daycon/test_pred.csv", index=False, encoding="utf-8")
    print("Decoded results saved to test_pred.csv.")
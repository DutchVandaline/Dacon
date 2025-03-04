import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AdamW
from Deobfuscate_text import deobfuscate_text
from Dataset import ObfuscationDataset
from train_step import train_step
from test_step import test_step
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10

tokenizer = GPT2TokenizerFast.from_pretrained("skt/kogpt2-base-v2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

model.resize_token_embeddings(len(tokenizer))
if model.lm_head.bias is not None:
    old_bias = model.lm_head.bias.data
    new_bias = torch.zeros((len(tokenizer),), device=old_bias.device)
    new_bias[:old_bias.size(0)] = old_bias
    model.lm_head.bias = torch.nn.Parameter(new_bias)
print("Model embedding layer resized.")

train_df = pd.read_csv(r"/home/junha/Daycon/Dataset/train.csv", encoding="utf-8")
train_dataset = ObfuscationDataset(train_df, tokenizer, max_length=512)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)

val_df = pd.read_csv(r"/home/junha/Daycon/Dataset/val.csv", encoding="utf-8")
val_dataset = ObfuscationDataset(train_df, tokenizer, max_length=512)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=True,
)

for epoch in tqdm(range(1, epochs + 1)):
    print(f"\nEpoch {epoch}/{epochs}")
    train_loss = train_step(model, train_dataloader, optimizer, device)
    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
    test_loss = test_step(model, val_dataloader, device)
    print(f"Epoch {epoch} - Test Loss: {test_loss:.4f}")
    model.save_pretrained(f"/home/junha/Daycon/Workstation/Models/kogpt2_epoch_{epoch}")
    tokenizer.save_pretrained(f"/home/junha/Daycon/Workstation/Models/kogpt2_epoch_{epoch}")
    print(f"Model saved after Epoch {epoch}.")
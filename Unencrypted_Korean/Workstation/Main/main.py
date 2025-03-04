import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from train_step import train_step
from test_step import test_step
from DaconDataset import DayconDataset
from Tokenizer import CharTokenizer
from collate_fn import collate_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv_path = "/home/junha/Daycon/Dataset/train.csv"
test_csv_path = "/home/junha/Daycon/Dataset/val.csv"
hangul_csv_path = "/home/junha/Daycon/Dataset/All_Possible_Korean_Syllables.csv"

epochs = 40
batch_size = 4
learning_rate = 1e-4

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

train_inputs = train_df["input"].tolist()
train_outputs = train_df["output"].tolist()
test_inputs = test_df["input"].tolist()
test_outputs = test_df["output"].tolist()

# Load SKT KoGPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
model.to(device)

train_input_ids = [torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long) for text in train_inputs]
train_target_ids = [torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long) for text in train_outputs]
test_input_ids = [torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long) for text in test_inputs]
test_target_ids = [torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long) for text in test_outputs]

train_dataset = DayconDataset(train_input_ids, train_target_ids)
test_dataset = DayconDataset(test_input_ids, test_target_ids)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    train_loss = train_step(model, train_dataloader, optimizer, device)
    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
    test_loss = test_step(model, test_dataloader, device)
    print(f"Epoch {epoch} - Test Loss: {test_loss:.4f}")
    model.save_pretrained(f"/home/junha/Daycon/Workstation/Models/kogpt2_epoch_{epoch}")
    tokenizer.save_pretrained(f"/home/junha/Daycon/Workstation/Models/kogpt2_epoch_{epoch}")
    print(f"Model saved after Epoch {epoch}.")

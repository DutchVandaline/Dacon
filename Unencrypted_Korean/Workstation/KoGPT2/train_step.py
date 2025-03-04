import torch

def train_step(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Train Loss: {avg_loss}")

    return avg_loss

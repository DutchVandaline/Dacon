import torch

def test_step(model, test_dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss}")

    return avg_loss

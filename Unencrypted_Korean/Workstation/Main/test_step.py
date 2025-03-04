import torch

def test_step(model, test_dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_tensor, target_tensor in test_dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            outputs = model(input_ids=input_tensor, labels=target_tensor)
            loss = outputs.loss

            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss}")

    return avg_loss

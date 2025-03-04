import torch

def train_step(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for input_tensor, target_tensor in train_dataloader:
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

        # 모델 실행
        outputs = model(input_ids=input_tensor, labels=target_tensor)
        loss = outputs.loss

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Train Loss: {avg_loss}")

    return avg_loss

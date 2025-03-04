import torch
import torch.nn

def train_step(model, dataloader, loss_fn, optimizer, device, accumulation_steps=4):
    model.train()
    train_running_loss = 0.0
    train_correct = 0
    total_train_samples = 0

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        train_running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = train_running_loss / len(dataloader)
    train_accuracy = train_correct / total_train_samples
    return avg_train_loss, train_accuracy

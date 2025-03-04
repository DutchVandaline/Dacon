import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import transforms, datasets
from torchvision.models.vision_transformer import vit_b_16
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

from Test_Step import test_step
from Train_Step import train_step

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "C:/junha/Git/Dacon/ImageClassification/Dataset"
submission_csv_path = "submission.csv"
LEARNING_RATE = 1e-5

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder(root=os.path.join(root_dir, "train"), transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root=os.path.join(root_dir, "val"), transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Dataloader Process Completed")

class_names = train_dataset.classes
num_classes = len(class_names)

model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

# 학습 및 검증 루프
num_epochs = 10
accumulation_steps = 4

for epoch in tqdm(range(num_epochs)):
    print(f"Epoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device, accumulation_steps)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    val_loss, val_acc, val_f1 = test_step(model, val_dataloader, loss_fn, device)
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1 Score: {val_f1:.4f}")

print("Training Completed!")

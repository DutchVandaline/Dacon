import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from ViT_Models.ViT_Anomaly_Detection import ViT_Anomaly_Detection

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters used during training
IMG_SIZE = 224
PATCH_SIZE = 16
EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 12
D_FF = 1536
DROP_OUT = 0.0
NUM_CLASSES = 2

model = ViT_Anomaly_Detection(
    img_size=IMG_SIZE,
    in_channels=3,
    patch_size=PATCH_SIZE,
    num_transformer_layers=NUM_LAYERS,
    embedding_dim=EMBEDDING_DIM,
    mlp_size=D_FF,
    num_heads=NUM_HEADS,
    mlp_dropout=DROP_OUT,
    embedding_dropout=DROP_OUT,
    mode="classification",
).to(device)

checkpoint_path = "/home/junha/Anomaly_Detection/CheckPoint/VanilaViT/VanillaViT_100.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

df = pd.read_csv("test.csv")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def inference(model, file_paths, transform, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: File {path} does not exist.")
                predictions.append(-1)
                continue

            image = Image.open(path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0).to(device)

            output = model(image)
            _, pred = torch.max(output, dim=1)
            predictions.append(pred.item())

    return predictions

image_paths = df["clean_output"].tolist()
preds = inference(model, image_paths, val_transform, device)

df["prediction"] = preds

label_map = {0: "Normal", 1: "Abnormal", -1: "FileNotFound"}
df["prediction_label"] = df["prediction"].map(label_map)

output_csv_path = "test_prediction.csv"
df.to_csv(output_csv_path, index=False)
print(f"Prediction saved to {output_csv_path}")

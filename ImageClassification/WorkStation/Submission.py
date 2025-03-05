import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.vision_transformer import vit_l_32
from PIL import Image
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_images_dir = r"C:\junha\Git\Dacon\ImageClassification\Dataset\test"

class_names = ["airplane", "apple", "ball", "bird", "building", "cat", "emoticon", "police_car", "rabbit", "truck"]
num_classes = len(class_names)

model = vit_l_32(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)

model_weights_path = r"C:\junha\Git\Dacon\ImageClassification\CheckPoint\ViT_pretrain_40.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.eval()

predictions = []
for filename in os.listdir(test_images_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_id = os.path.splitext(filename)[0]
        image_path = os.path.join(test_images_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {image_path}: {e}")
            continue
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, pred_idx = torch.max(outputs, 1)
        predicted_label = class_names[pred_idx.item()]
        predictions.append({"ID": image_id, "label": predicted_label})

df_submission = pd.DataFrame(predictions)
df_submission = df_submission.sort_values("ID")
submission_csv_path = "../submission.csv"
df_submission.to_csv(submission_csv_path, index=False)

print("Submission file saved to:", submission_csv_path)
print(df_submission.head())

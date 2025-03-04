import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
from PIL import Image

# =======================================
# 1) 경로 설정
# =======================================
train_data_dir = r"C:\junha\Daycon\ImageClassification\Dataset\train"
test_data_dir = r"C:\junha\Daycon\ImageClassification\Dataset\test"
submission_csv_path = "submission.csv"  # CSV 출력 파일 경로

# =======================================
# 2) 데이터 전처리 정의
# =======================================
# 주의: Grayscale 이미지를 ViT에 입력하기 위해선 3채널로 맞춰줘야 합니다.
#       학습 시 사용한 것과 동일해야 합니다.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 원래 3채널이면 mean/std를 [0.5, 0.5, 0.5] 등으로 설정하는 것이 일반적이나
    # 사용자가 기존에 mean=[0.5], std=[0.5]를 사용하셨다면 그대로 유지합니다.
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =======================================
# 3) 데이터셋 & DataLoader 준비
# =======================================
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 테스트 데이터도 ImageFolder 구조를 가정
# (ex. grayscale_images_test/airplane/*.jpg, grayscale_images_test/automobile/*.jpg ...)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =======================================
# 4) 모델 준비 (사전학습된 ViT + head 교체)
# =======================================
num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)

# =======================================
# 5) 손실 함수 & 옵티마이저
# =======================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =======================================
# 6) 학습 루프
# =======================================
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Training Finished!\n")

# =======================================
# 7) 테스트 데이터셋 평가
# =======================================
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

# =======================================
# 8) 테스트 디렉터리 전체 이미지 분류 후 CSV 저장
# =======================================
# 테스트 폴더 내부 구조가 ImageFolder 형태가 아닐 수도 있으므로,
# "단순히 폴더 내 모든 이미지를 대상으로 예측"한다면 아래 로직 사용
# (ImageFolder 구조를 그대로 사용해도 되지만, 요구사항에 맞게 변경)

print("Generating submission.csv ...")
with open(submission_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "label"])  # 헤더

    # test_data_dir 안에 있는 모든 이미지 파일에 대해
    for filename in os.listdir(test_data_dir):
        # ImageFolder 구조를 사용하지 않고, 단순히 이미지 확장자로 필터링
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # 파일 경로
            image_path = os.path.join(test_data_dir, filename)
            # ID는 파일명(확장자 제외)으로 가정
            image_id = os.path.splitext(filename)[0]

            # 이미지 로드 & 전처리
            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            # 예측
            with torch.no_grad():
                output = model(img)
                _, pred = torch.max(output, 1)

            # 예측 클래스 이름
            predicted_label = train_dataset.classes[pred.item()]

            # CSV에 쓰기
            writer.writerow([image_id, predicted_label])

print(f"Done! CSV saved at: {submission_csv_path}")

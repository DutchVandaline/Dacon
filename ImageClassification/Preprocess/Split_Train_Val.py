import os
import shutil
import random


train_dir = "C:/junha/Git/Dacon/ImageClassification/Dataset/train"
val_dir = "C:/junha/Git/Dacon/ImageClassification/Dataset/val"

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        val_class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) == 0:
            continue

        num_val = max(1, int(len(images) * 0.2))
        val_images = random.sample(images, num_val)

        # validation 디렉토리로 이동 (원본은 제거)
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(val_class_dir, img)
            shutil.move(src_path, dst_path)

        print(f"Class '{class_name}': {num_val} images moved to validation folder.")

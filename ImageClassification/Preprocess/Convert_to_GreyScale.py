import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# CSV 파일 로드
file_path = "C:/junha/Datasets/Daycon/image/open/test.csv"
df = pd.read_csv(file_path)

# 이미지 저장 디렉토리 생성
output_dir = "C:/junha/Daycon/ImageClassification/Dataset/test"
os.makedirs(output_dir, exist_ok=True)

# 픽셀 데이터 추출
pixel_data = df.iloc[:, 1:].values  # ID와 label을 제외한 픽셀 데이터만 사용

# 32x32 그레이스케일 이미지 변환 및 저장
for i, row in enumerate(pixel_data):
    image_array = row.reshape(32, 32).astype(np.uint8)  # 1채널 그레이스케일
    image = Image.fromarray(image_array, mode='L')  # 'L' 모드는 8-bit grayscale
    image.save(os.path.join(output_dir, f"{df.iloc[i, 0]}.png"))

print(f"이미지 {len(pixel_data)}개가 {output_dir}에 저장되었습니다.")

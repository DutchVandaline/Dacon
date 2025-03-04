import pandas as pd
import numpy as np
import os
from PIL import Image

# CSV 파일 로드
file_path = "C:/junha/Datasets/Daycon/image/open/test.csv"
df = pd.read_csv(file_path)

# 기본 이미지 저장 디렉토리 생성
base_output_dir = f"C:\junha\Daycon\ImageClassification\Dataset\grayscale_images/"
os.makedirs(base_output_dir, exist_ok=True)

# 클래스별 디렉토리 생성
class_dirs = {}
for class_name in df['label'].unique():
    class_dir = os.path.join(base_output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    class_dirs[class_name] = class_dir

# 픽셀 데이터 추출
pixel_data = df.iloc[:, 2:].values  # ID와 label을 제외한 픽셀 데이터만 사용

# 32x32 그레이스케일 이미지 변환 및 클래스별 저장
for i, row in enumerate(pixel_data):
    image_array = row.reshape(32, 32).astype(np.uint8)  # 1채널 그레이스케일
    image = Image.fromarray(image_array, mode='L')  # 'L' 모드는 8-bit grayscale

    class_name = df.iloc[i, 1]  # 해당 이미지의 클래스명
    image_path = os.path.join(class_dirs[class_name], f"{df.iloc[i, 0]}.png")
    image.save(image_path)

print(f"이미지 {len(pixel_data)}개가 클래스별 디렉토리에 저장되었습니다.")

import pandas as pd
from sklearn.model_selection import train_test_split

# CSV 파일 경로
train_csv_path = "C:/junha/Datasets/Daycon/new_train_preprocessed.csv"

# Train 데이터 불러오기
df = pd.read_csv(train_csv_path)

# 데이터 확인
print(f"전체 데이터 개수: {len(df)}")

# 80%: Train, 20%: Validation 데이터로 분할
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 분할된 데이터 크기 확인
print(f"Train 데이터 개수: {len(train_df)}")
print(f"Validation 데이터 개수: {len(val_df)}")

# 새로운 CSV 파일 저장
train_df.to_csv("C:/junha/Datasets/Daycon/new_train.csv", index=False)
val_df.to_csv("C:/junha/Datasets/Daycon/val.csv", index=False)

print("✅ Train & Validation 데이터 분할 완료!")

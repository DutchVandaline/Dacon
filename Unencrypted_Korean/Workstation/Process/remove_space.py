import pandas as pd

# CSV 파일 불러오기
train_df = pd.read_csv("C:/junha/Datasets/Daycon/new_train.csv")
val_df   = pd.read_csv("C:/junha/Datasets/Daycon/val.csv")

# 정규표현식: 한글(가-힣), 영어(대소문자), 숫자(0-9), .,!? 를 제외한 모든 문자 제거
pattern = r'[^가-힣a-zA-Z0-9\.\,\!\?]'

# train_df의 input, output에서 위 패턴에 해당하는 문자 제거
train_df["input"]  = train_df["input"].str.replace(pattern, "", regex=True)
train_df["output"] = train_df["output"].str.replace(pattern, "", regex=True)

# val_df의 input, output에서도 동일 처리
val_df["input"]  = val_df["input"].str.replace(pattern, "", regex=True)
val_df["output"] = val_df["output"].str.replace(pattern, "", regex=True)

# 수정된 DataFrame을 다시 CSV 파일로 저장
train_df.to_csv("C:/junha/Datasets/Daycon/new_train.csv", index=False)
val_df.to_csv("C:/junha/Datasets/Daycon/val.csv", index=False)

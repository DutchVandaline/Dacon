from collections import defaultdict
import pandas as pd

def build_correction_map(train_df):
    """
    학습 데이터에서 난독화 패턴을 분석하여 문자 치환 맵을 생성
    """
    correction_map = defaultdict(str)
    for _, row in train_df.iterrows():
        input_text = row["input"]
        output_text = row["output"]
        for in_char, out_char in zip(input_text, output_text):
            if in_char != out_char:
                correction_map[in_char] = out_char
    return correction_map

def correct_text(text, correction_map):
    return "".join(correction_map.get(char, char) for char in text)

def apply_deobfuscation(input_csv, output_csv, correction_map):
    """
    입력 CSV 파일을 변환하여 새로운 CSV 파일로 저장
    """
    df = pd.read_csv(input_csv)
    df["clean_output"] = df["input"].apply(lambda x: correct_text(x, correction_map))
    df.to_csv(output_csv, index=False)
    print(f"Deobfuscated file saved to: {output_csv}")

if __name__ == "__main__":
    train_df = pd.read_csv("C:/junha/Datasets/Daycon/train.csv")
    correction_map = build_correction_map(train_df)

    # test.csv에 적용하여 변환된 파일 저장
    apply_deobfuscation("C:/junha/Datasets/Daycon/test.csv", "C:/junha/Datasets/Daycon/test_preprocessed.csv", correction_map)

import pandas as pd


def remove_columns_from_csv(input_file, output_file, columns_to_remove):
    """
    CSV 파일에서 특정 열을 제거하는 함수
    :param input_file: 입력 CSV 파일 경로
    :param output_file: 출력 CSV 파일 경로
    :param columns_to_remove: 제거할 열의 리스트
    """
    # CSV 파일 읽기
    df = pd.read_csv(input_file)

    # 특정 열 제거
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    # 수정된 데이터 저장
    df.to_csv(output_file, index=False)

    print(f"Columns {columns_to_remove} removed and saved to {output_file}")


# 사용 예제
input_csv = "C:/junha/Datasets/Daycon/prediction.csv"
output_csv = "output.csv"
columns_to_remove = ["input", "clean_output"]  # 제거할 열 목록

remove_columns_from_csv(input_csv, output_csv, columns_to_remove)

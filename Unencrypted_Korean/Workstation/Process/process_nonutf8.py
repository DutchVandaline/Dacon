import pandas as pd
import unicodedata
import re


def normalize_hangul_jamo(text):
    """
    Normalize Unicode and replace isolated Hangul Jamo (ᄏ → ㅋ, ᄒᄒ → ㅎㅎ)
    """
    if pd.isna(text):  # Handle NaN values
        return text

    # Normalize Unicode (fix decomposed Hangul like ᄏᄏᄏ to ㅋㅋㅋ)
    text = unicodedata.normalize('NFC', text)

    # Replace isolated Hangul Jamo with full Hangul characters
    text = text.replace("ᄏ", "ㅋ").replace("ᄒ", "ㅎ")

    return text


def remove_newlines(text):
    """
    Replace newline characters with a space
    """
    if pd.isna(text):
        return text
    return text.replace('\n', ' ')


# Load the CSV file
def clean_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    if 'decoded' in df.columns:
        df['decoded'] = df['decoded'].apply(normalize_hangul_jamo)

    if 'output' in df.columns:
        df['output'] = df['output'].apply(remove_newlines)

    df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to: {output_csv}")
# Example usage
if __name__ == "__main__":
    input_csv_path = "C:/junha/Datasets/Daycon/prediction.csv"
    output_csv_path = "C:/junha/Datasets/Daycon/prediction_cleaned.csv"
    clean_csv(input_csv_path, output_csv_path)
